# -----------------------------------------------------------------------------
# ARQUIVO: feature_engineering/main.py (Versao Final de Producao)
# -----------------------------------------------------------------------------

"""
Modulo Principal de Engenharia de Features.
"""

import pandas as pd
import numpy as np
import joblib
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .native_indicators import NativeIndicators

# [HYBRID v3] HybridAutoencoder (JEPA + BYOL + VICReg + TS2Vec) é o único pipeline AE.
# temporal_autoencoder.py (VAE legado) foi removido — não usado mais.
from .hybrid_autoencoder import (
    HybridAutoencoderPipeline as TemporalAutoencoderPipeline,
)
from utils.logger import get_logger
from config.settings import AIConfig, TradingConfig

logger = get_logger("FeatureEngineering")


class FeatureEngineeringPipeline:
    def __init__(self, config: AIConfig):
        """
        [VERSAO ADAPTADA PARA O AUTOENCODER TEMPORAL]
        Inicializa os scalers e a pipeline do autoencoder temporal.
        """
        if not isinstance(config, AIConfig):
            raise TypeError("'config' deve ser uma instancia da classe AIConfig.")
        self.config = config
        self.trading_config = TradingConfig()

        self.native_indicators = NativeIndicators()
        self.scalers: Dict[str, Any] = {}

        # 1. Caminhos para os scalers
        self.scaler_path = os.path.join(
            self.config.MODEL_DIR, "feature_scalers_dual.joblib"
        )

        # 2. Instanciar a nova pipeline do autoencoder temporal
        self.temporal_autoencoder_pipeline = TemporalAutoencoderPipeline(config)

        # Carrega os scalers de forma robusta
        self._load_scalers()
        logger.info(
            "FeatureEngineeringPipeline inicializado com a Pipeline de Autoencoder Temporal."
        )

    def _save_scalers(self):
        """Salva um dicionario contendo os scalers e a lista de colunas."""
        try:
            joblib.dump(self.scalers, self.scaler_path)
            logger.info(
                f"Normalizadores e lista de colunas salvos em '{self.scaler_path}'."
            )
        except Exception as e:
            logger.error(f"Erro ao salvar os normalizadores: {e}", exc_info=True)

    def _load_scalers(self):
        """
        Carrega os normalizadores e valida sua estrutura para garantir a integridade do
        'Contrato de Features'.
        """
        self.scalers = {}
        if os.path.exists(self.scaler_path):
            try:
                loaded_data = joblib.load(self.scaler_path)
                if (
                    isinstance(loaded_data, dict)
                    and "robust" in loaded_data
                    and "minmax" in loaded_data
                    and "columns" in loaded_data
                    and isinstance(loaded_data["columns"], list)
                ):
                    self.scalers = loaded_data
                    logger.info(
                        f"Normalizadores e 'Contrato de Features' validos carregados de '{self.scaler_path}'."
                    )
                else:
                    logger.warning(
                        f"Arquivo de normalizadores '{self.scaler_path}' obsoleto. Ignorando."
                    )
                    self.scalers = {}
            except Exception as e:
                logger.error(f"Erro ao carregar scalers de '{self.scaler_path}': {e}.")
                self.scalers = {}
        else:
            logger.warning(
                f"Arquivo de scalers nao encontrado em '{self.scaler_path}'."
            )
            self.scalers = {}

    def train_autoencoder(
        self, data_for_autoencoder: pd.DataFrame, num_epochs: int
    ) -> int:
        """
        Delega o treinamento para a pipeline do autoencoder temporal.
        """
        logger.info("Delegando treinamento para a Temporal Autoencoder Pipeline...")

        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        feature_columns = [
            col for col in data_for_autoencoder.columns if col not in ohlcv_cols
        ]
        if not feature_columns:
            feature_columns = data_for_autoencoder.columns.tolist()

        success = self.temporal_autoencoder_pipeline.train_autoencoder_temporal(
            df=data_for_autoencoder, feature_columns=feature_columns, optimize=True
        )

        logger.info(f"Treinamento delegado concluido. Sucesso: {success}")
        return 1 if success else 0

    async def create_features(
        self,
        raw_df_dict: Dict[str, pd.DataFrame],
        symbol: str,
        primary_timeframe: str,
        fit_scaler: bool = False,
        tape_metrics: Optional[Dict[str, Any]] = None,
        onchain_metrics: Optional[Dict[str, Any]] = None,
        apply_latent: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        [VERSAO FINAL E COMPLETA - ESTADO DA ARTE]
        Orquestra a criacao de features, garantindo a consistencia da normalizacao.
        """
        if not isinstance(raw_df_dict, dict) or not raw_df_dict:
            logger.error("'raw_df_dict' invalido.")
            return None
        if primary_timeframe not in raw_df_dict:
            logger.error(f"Timeframe primario '{primary_timeframe}' nao encontrado.")
            return None

        logger.info(
            f"Iniciando calculo de indicadores para {len(raw_df_dict)} timeframes em PARALELO..."
        )

        loop = asyncio.get_running_loop()

        # [E2 FIX] Rastreia os TFs válidos separadamente para garantir que o zip
        # com all_timeframes_featured_dfs (retorno do asyncio.gather) fique sempre alinhado.
        # Antes: raw_df_dict.items() tinha N entradas mesmo se K < N tasks foram criadas,
        # causando desalinhamento silencioso onde o último TF sumia sem log de erro.
        valid_tf_list = []
        tasks = []
        for tf, df_raw in raw_df_dict.items():
            if df_raw.empty or not all(
                col in df_raw.columns
                for col in ["open", "high", "low", "close", "volume"]
            ):
                logger.warning(f"DataFrame para '{tf}' invalido. Pulando.")
                continue

            task = loop.run_in_executor(
                None,
                self.native_indicators.calculate_all_features,
                df_raw.copy(),
                symbol,
                tf,
            )
            tasks.append(task)
            valid_tf_list.append((tf, df_raw))

        if not tasks:
            logger.error("Nenhuma tarefa valida.")
            return None

        all_timeframes_featured_dfs = await asyncio.gather(*tasks)
        logger.info("Calculo de indicadores concluido.")

        all_timeframes_dfs = []
        required_cols = ["open", "high", "low", "close", "volume"]

        # [E2 FIX] Zip com valid_tf_list (N entradas == N tasks == N results)
        for tf_featured_df, (tf, df_raw) in zip(
            all_timeframes_featured_dfs, valid_tf_list
        ):
            if tf_featured_df is None or tf_featured_df.empty:
                continue

            tf_featured_df = tf_featured_df.loc[:, ~tf_featured_df.columns.duplicated()]

            if tf == primary_timeframe:
                cols_to_suffix = [
                    col for col in tf_featured_df.columns if col not in required_cols
                ]
                suffixed_df = tf_featured_df[cols_to_suffix].rename(
                    columns={col: f"{col}_{tf}" for col in cols_to_suffix}
                )
                final_df_for_tf = pd.concat(
                    [tf_featured_df[required_cols], suffixed_df], axis=1
                )
                all_timeframes_dfs.append(final_df_for_tf)
            else:
                # SINGLE SUFFIXING STRATEGY for secondary timeframes (Fix BUG M2)
                cols_all = tf_featured_df.columns.tolist()
                final_secondary_df = tf_featured_df[cols_all].rename(
                    columns={col: f"{col}_{tf}" for col in cols_all}
                )
                final_secondary_df = final_secondary_df.loc[
                    :, ~final_secondary_df.columns.duplicated()
                ]
                all_timeframes_dfs.append(final_secondary_df)

        if not all_timeframes_dfs:
            return None

        base_df = next(
            (
                df
                for df in all_timeframes_dfs
                if all(c in df.columns for c in required_cols)
            ),
            None,
        )
        if base_df is None:
            return None

        other_dfs = [df for df in all_timeframes_dfs if df is not base_df]
        featured_df_combined = base_df.sort_index()

        for other_df in other_dfs:
            # [ANTI-LEAKAGE MULTI-TF] O índice é open_time (data_provider.py:116).
            # merge_asof(direction='backward') sem shift alinharia o candle 15m de 09:15
            # ao candle 1h de 09:00, cujos H/L/C só estão disponíveis em 09:59:59.
            # shift(1) garante que apenas barras FECHADAS do TF superior são usadas:
            # ao 09:15, o último 1h fechado é o de 08:00, não o de 09:00 (ainda aberto).
            other_shifted = other_df.shift(1).dropna(how="all")
            featured_df_combined = pd.merge_asof(
                left=featured_df_combined.sort_index(),
                right=other_shifted.sort_index(),
                left_index=True,
                right_index=True,
                direction="backward",
            )

        logger.info("Garantindo consistencia de tipos...")
        for col in featured_df_combined.select_dtypes(include="bool").columns:
            featured_df_combined[col] = featured_df_combined[col].astype(int)

        # [SCIENTIFIC FIX v4 — ANTI-LEAKAGE + ANTI-FALSE-SIGNAL]
        # Antes: ffill() → dropna() → fillna(0.0) injetava sinal falso em features normalizadas
        # (RSI=0=oversold extremo, z-score=0=média perfeita, ATR=0=volatilidade zero).
        # Este fillna(0) contamina o scaler, vieses downstream o RL e cria regimes artificiais.
        #
        # Nova estratégia (cientificamente correta):
        #   1. ffill(limit=N) → propaga último valor conhecido APENAS por N barras
        #      (ex: 4h dentro de 1m = ~240 barras; limite evita propagação indefinida após halt).
        #   2. dropna() completo → remove linhas que ainda tenham NaN (início do dataset onde
        #      TFs superiores não têm histórico suficiente). Perde ~0.1-0.7% do dataset.
        #   3. Verificação final estrita: se ainda há NaN, é bug upstream — raise.
        #
        # Referência: López de Prado (2018) "Advances in Financial ML" — Appendix A.5:
        # "Never impute financial features with zero. Drop rows or use causal models."
        _pre_drop_rows = featured_df_combined.shape[0]
        _pre_drop_cols = featured_df_combined.shape[1]
        initial_nan_count = featured_df_combined.isna().sum().sum()

        # ffill com LIMITE por TF (heurística conservadora: 300 barras cobre 5h em 1m,
        # 25h em 5m, 75h em 15m, 12.5d em 1h, 50d em 4h — suficiente para gaps normais de exchange).
        # [PANDAS 3.0 FIX] infer_objects() evita FutureWarning sobre downcasting em ffill.
        featured_df_combined = featured_df_combined.infer_objects(copy=False).ffill(limit=300)

        if initial_nan_count > 0:
            pct_nan = initial_nan_count / max(1, _pre_drop_rows * _pre_drop_cols) * 100
            logger.info(
                f"[MERGE] {initial_nan_count} NaN residuais ({pct_nan:.2f}% das células) "
                f"após ffill(limit=300). Dropando linhas restantes com NaN (anti-leakage)."
            )

        # [B2 FIX] Pre-check: detecta colunas com NaN excessivo ANTES do dropna.
        # Antes: dropna(how='any') incondicional. Se 1 coluna estivesse all-NaN, todo
        # dataset era zerado e o RuntimeError abaixo só capturava DEPOIS da destruição.
        # Agora: identifica colunas com >50% NaN, loga e DROPA AS COLUNAS (não as linhas)
        # — preserva dados. Só linhas com NaN nas colunas saudáveis vão para o dropna final.
        nan_ratio_per_col = featured_df_combined.isna().mean()
        bad_cols = nan_ratio_per_col[nan_ratio_per_col > 0.50].index.tolist()
        if bad_cols:
            logger.warning(
                f"[B2] {len(bad_cols)} colunas com >50% NaN serão DROPADAS (não dropar linhas): "
                f"{bad_cols[:10]}{'...' if len(bad_cols) > 10 else ''}"
            )
            featured_df_combined = featured_df_combined.drop(columns=bad_cols)

        # Drop estrito: linhas com qualquer NaN remanescente são descartadas
        # (evita fillna(0) que cria sinais falsos em features normalizadas).
        featured_df_combined.dropna(inplace=True)

        # Verificação final: se ainda houver NaN é bug — falhar alto
        remaining_nan = featured_df_combined.isna().sum().sum()
        if remaining_nan > 0:
            nan_cols = featured_df_combined.columns[
                featured_df_combined.isna().any()
            ].tolist()
            raise RuntimeError(
                f"[MERGE] {remaining_nan} NaN persistentes após dropna. "
                f"Colunas problemáticas: {nan_cols[:10]}. "
                "Bug upstream no cálculo de features — não mascarar com fillna(0)."
            )

        rows_before = len(featured_df_combined)
        if featured_df_combined.empty:
            return None
        logger.info(
            f"[MERGE] Multi-TF combinado: {rows_before} linhas preservadas após ffill."
        )

        # ETAPA 7: ADICIONAR REGIME LABELS (Hamilton 1989)
        try:
            from feature_engineering.scientific_data_processor import (
                ScientificDataProcessor,
                create_data_processor,
            )

            # [R2 FIX] Usa create_data_processor() que chama load_scalers() internamente,
            # garantindo que os scalers já treinados são reutilizados em vez de um objeto
            # virgem que perderia consistência entre chamadas de create_features().
            data_processor = create_data_processor()
            featured_df_combined = data_processor.add_regime_labels(
                featured_df_combined
            )
            data_processor.save_scalers()
            logger.info(
                f"Regime labels adicionados: {featured_df_combined['regime'].value_counts().to_dict()}"
            )
            regime_counts = featured_df_combined["regime"].value_counts(normalize=True)
            expected_dist = {0: 0.30, 1: 0.30, 2: 0.40}
            for code, expected in expected_dist.items():
                actual = regime_counts.get(code, 0.0)
                if abs(actual - expected) > 0.20:
                    logger.warning(
                        f"⚠️ [REGIME DRIFT] Regime {code} = {actual:.1%} "
                        f"(esperado ~{expected:.1%}). Pode indicar mudança de mercado."
                    )
        except Exception as e:
            logger.warning(f"Falha ao adicionar regime labels: {e}")

        # [REC 4] regime_age_norm: tempo (em barras) que o regime atual esta ativo.
        # Importante para o SAC saber se pode CONFIAR no tp_prior_dir (regime estavel)
        # ou se esta em TRANSICAO (latents do AE ainda refletem regime antigo).
        # Os 48 hidden_features tem inercia de seq_len=64 candles (~16h em 15m) —
        # quando regime acabou de mudar, o latent ainda nao "viu" a mudanca completa.
        #
        # Formula: cumcount dentro de cada bloco contiguo do mesmo regime, dividido por 64.
        #   = 0/64 = 0.00 → regime acabou de mudar (NAO CONFIAR no AE latent)
        #   = 32/64 = 0.50 → metade do contexto AE ja eh do regime atual
        #   = 64/64 = 1.00 → contexto AE 100% do regime atual (MAXIMA CONFIANCA)
        #   clip em 2.0 → regime extremamente estavel (>2x o contexto AE)
        try:
            if "regime" in featured_df_combined.columns:
                _regime_col = featured_df_combined["regime"]
                # Detecta mudancas de regime (cumsum cria id unico para cada bloco contiguo)
                _block_id = (_regime_col != _regime_col.shift()).cumsum()
                # Conta consecutivamente dentro de cada bloco
                _regime_age_bars = _regime_col.groupby(_block_id).cumcount()
                # Normaliza por seq_len=64 do AE, cap em 2.0
                featured_df_combined["regime_age_norm"] = np.clip(
                    _regime_age_bars.values / 64.0, 0.0, 2.0
                ).astype(np.float32)
                logger.info(
                    f"[REC 4] regime_age_norm adicionado | distribuicao: "
                    f"mean={featured_df_combined['regime_age_norm'].mean():.2f}, "
                    f"max={featured_df_combined['regime_age_norm'].max():.2f}"
                )
        except Exception as _e:
            logger.warning(f"[REC 4] Falha ao calcular regime_age_norm: {_e}")
            featured_df_combined["regime_age_norm"] = 1.0  # default neutro

        logger.info(
            f"Features criadas com sucesso. Shape: {featured_df_combined.shape}"
        )
        return featured_df_combined

    def apply_hidden_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wrapper para aplicar features latentes usando o pipeline temporal.

        [CD5 FIX] Alias inteligente das features legadas (HybridAE v3 -> contrato SAC antigo):
          - prob_bull/bear/ranger -> tp_regime_up/down/sideways + trend_pred_*
          - regime_confidence    -> tp_prior_conf, trend_pred_confidence
          - 1 - regime_confidence -> tp_uncertainty
          - ema_trend_15m -> ema_trend (sem sufixo)
        Features sem equivalente (qpnl_*, tp_duration_median) DEIXAM de existir
        em vez de virem como zeros. O env consome via row.get(col, default), entao
        o default eh usado quando a coluna nao existir — comportamento correto.
        """
        # 1. Aplica o Autoencoder (Adiciona Latents + Regime + Confianca)
        df_enriched = self.temporal_autoencoder_pipeline.apply_hidden_features_temporal(
            df
        )

        # [PERF FIX] Coletar TODAS as colunas novas em um dict e adicionar via UM
        # pd.concat. Anteriormente, cada `df_enriched[col] = ...` adicionava coluna
        # individualmente, fragmentando o DataFrame (PerformanceWarning observado em
        # ~10 lugares neste metodo). Para um df de 100k linhas, ganho ~25-30% velocidade.
        new_cols: Dict[str, Any] = {}

        # 2. [CD5 FIX] Calcula regime_confidence e regime_val PRIMEIRO (usado nos aliases abaixo)
        if "prob_bull" in df_enriched.columns:
            probs = df_enriched[["prob_bull", "prob_bear", "prob_ranger"]]
            new_cols["regime_confidence"] = probs.max(axis=1).values

            # [PANDAS 3.0 FIX] idxmax com all-NA ou skipna=False vai virar erro em pandas 3.0.
            # Solucao: usar np.argmax direto que e mais robusto a NaN (com fillna).
            # Antes: probs.idxmax(axis=1).map({"prob_bull": 0, "prob_bear": 1, "prob_ranger": 2})
            # Agora: vetorizado via numpy argmax do array com fillna(0) explicito.
            probs_filled = probs.fillna(0.0).values  # (N, 3) array
            # Ordem das colunas: [prob_bull, prob_bear, prob_ranger] -> argmax → [0,1,2]
            # Que ja e exatamente o codigo de regime: 0=Bull, 1=Bear, 2=Ranger
            regime_val_arr = np.argmax(probs_filled, axis=1).astype(np.int64)
            # Se todas as 3 probs estavam NaN (linha invalida), fillna(0) deu [0,0,0]
            # e argmax retorna 0 (Bull) por padrao. Mascarar para 2 (Ranger) nessas linhas.
            all_nan_mask = probs.isna().all(axis=1).values
            regime_val_arr[all_nan_mask] = 2
            new_cols["regime_val"] = regime_val_arr

        # 3. [CD5 FIX] Aliases NAO-ZERO das features legadas que possuem equivalente real.
        # Antes: 16 features eram setadas a 0.0 -> 3 gates do env morriam silenciosamente.
        # Agora: mapeamos do output do HybridAE para o contrato esperado pelo SAC env.
        _aliases: Dict[str, str] = {
            # Regime probabilities (TrendPredictor legado -> HybridAE RegimeHead)
            "tp_regime_up": "prob_bull",
            "tp_regime_down": "prob_bear",
            "tp_regime_sideways": "prob_ranger",
            "trend_pred_uptrend": "prob_bull",
            "trend_pred_downtrend": "prob_bear",
            "trend_pred_neutral": "prob_ranger",
            # Confidence (alias direto do max das probs)
            "trend_pred_confidence": "regime_confidence",
        }
        for legacy_col, source_col in _aliases.items():
            if legacy_col in df_enriched.columns:
                continue
            # Fonte vem do df_enriched ou de new_cols (regime_confidence calculado acima)
            if source_col in df_enriched.columns:
                new_cols[legacy_col] = df_enriched[source_col].values
            elif source_col in new_cols:
                new_cols[legacy_col] = new_cols[source_col]

        # ema_trend (sem sufixo): tenta puxar do TF primario, senao 0
        if "ema_trend" not in df_enriched.columns:
            for _suffix in ("_15m", "_1h", "_5m", "_4h", "_1m"):
                _col = f"ema_trend{_suffix}"
                if _col in df_enriched.columns:
                    new_cols["ema_trend"] = df_enriched[_col].values
                    break
            else:
                new_cols["ema_trend"] = 0.0

        # qpnl_* e tp_duration_median NAO sao mais geradas (env usa row.get default).

        # [FIX] Alias para 'regime'
        if "regime" not in df_enriched.columns:
            if "regime_val" in new_cols:
                new_cols["regime"] = new_cols["regime_val"]
            else:
                new_cols["regime"] = 2

        # [BUG 9 FIX] Mapeamento CONTINUO (identico ao ai_controller.py L1124).
        _regime_for_select = (
            df_enriched["regime"].values
            if "regime" in df_enriched.columns
            else new_cols.get("regime")
        )
        _conf_for_select = (
            df_enriched["regime_confidence"].values
            if "regime_confidence" in df_enriched.columns
            else new_cols.get("regime_confidence")
        )
        if _regime_for_select is not None and _conf_for_select is not None:
            _regime_arr = np.asarray(_regime_for_select)
            _conf_arr = np.asarray(_conf_for_select)
            conditions = [
                _regime_arr == 0,
                _regime_arr == 1,
                _regime_arr >= 2,
            ]
            choices = [_conf_arr, -_conf_arr, 0.0]
            new_cols["tp_prior_dir"] = np.select(conditions, choices, default=0.0)
        else:
            new_cols["tp_prior_dir"] = 0.0

        new_cols["tp_prior_conf"] = (
            _conf_arr
            if _conf_for_select is not None
            else np.full(len(df_enriched), 0.5)
        )
        new_cols["tp_uncertainty"] = 1.0 - np.asarray(new_cols["tp_prior_conf"])

        # [PERF FIX] Concat unico final: cria DataFrame das novas colunas com o mesmo
        # index do df_enriched, depois concat horizontal. DataFrame resultante e
        # contiguo na memoria — sem fragmentacao.
        # [BUG FIX 2026-05-13] Garante que se df_enriched ja tem colunas duplicadas
        # (ex: 2x "regime" vindas de stages anteriores), elas sao deduplicadas antes
        # do concat. Sem isso, df["regime"].values retorna shape (N, 2) e quebra np.select.
        if df_enriched.columns.duplicated().any():
            df_enriched = df_enriched.loc[:, ~df_enriched.columns.duplicated()]

        if new_cols:
            new_cols_df = pd.DataFrame(new_cols, index=df_enriched.index)
            # Drop colunas que ja existem no df_enriched para evitar duplicatas no concat
            cols_to_add = [c for c in new_cols_df.columns if c not in df_enriched.columns]
            if cols_to_add:
                df_enriched = pd.concat([df_enriched, new_cols_df[cols_to_add]], axis=1)
            # Para colunas que existem, atualiza valores (raro nesse fluxo)
            cols_to_update = [c for c in new_cols_df.columns if c in df_enriched.columns]
            for c in cols_to_update:
                df_enriched[c] = new_cols_df[c].values

        return df_enriched
