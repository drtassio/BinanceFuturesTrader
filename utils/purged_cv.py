# -----------------------------------------------------------------------------
# ARQUIVO: utils/purged_cv.py
# -----------------------------------------------------------------------------

"""
Purged Cross-Validation para Séries Temporais Financeiras.

Implementa técnicas de validação cruzada que respeitam:
1. Ordenação temporal (sem embaralhamento)
2. Embargo entre train/val (evita autocorrelação)
3. Purging de observações overlapping

Referência Acadêmica:
    De Prado, M. L. (2018). "Advances in Financial Machine Learning", Chapter 7:
    "Cross-Validation in Finance"
    
    Problema: CV padrão assume IID (independente e identicamente distribuído)
    Solução: Purged K-Fold CV remove observações que overlap temporalmente
"""

import warnings
from typing import Iterator, Tuple, Optional
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("PurgedCV")


class PurgedKFold:
    """
    K-Fold Cross-Validation com Purging e Embargo para dados temporais.
    
    Modificações vs sklearn.KFold:
    1. Mantém ordem temporal (no shuffle)
    2. Remove (purge) observações do treino que overlap com validação
    3. Adiciona embargo gap entre último treino e primeiro validação
    
    Exemplo:
        Timeline: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        Fold 1: Train=[0,1,2,3], Embargo=[4,5], Val=[6,7]
        Fold 2: Train=[0,1] + [8,9], Embargo=[6,7], Val=[4,5]
                ↑ Note: [2,3] purged (overlap com val anterior)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_bars: int = 10,
        purge_bars: int = 0
    ):
        """
        Args:
            n_splits: Número de folds
            embargo_bars: Barras de embargo após cada fold de treino
            purge_bars: Barras a remover (purge) antes de cada validação
                       Tipicamente = max(seq_length, label_lookahead)
        """
        if n_splits < 2:
            raise ValueError("n_splits deve ser >= 2")
        
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.purge_bars = purge_bars
        
        logger.info(
            f"📊 [PurgedCV] Configurado: {n_splits} splits, "
            f"embargo={embargo_bars}, purge={purge_bars}"
        )
    
    def split(
        self, 
        X: pd.DataFrame | np.ndarray,
        y: Optional[pd.Series | np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Gera índices (train, val) para cada fold.
        
        Args:
            X: Features (usado apenas para obter número de amostras)
            y: Labels (não usado, compatibilidade com sklearn)
            groups: Não usado
            
        Yields:
            (train_indices, val_indices) para cada fold
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calcula tamanho de cada fold
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1  # Distribui remainder
        
        # Calcula boundaries de cada fold
        fold_ends = np.cumsum(fold_sizes)
        fold_starts = np.concatenate([[0], fold_ends[:-1]])
        
        for fold_idx in range(self.n_splits):
            # Validação: bloco contíguo
            val_start = fold_starts[fold_idx]
            val_end = fold_ends[fold_idx]
            val_indices = indices[val_start:val_end]
            
            # Treino: tudo exceto validação e embargo
            train_mask = np.ones(n_samples, dtype=bool)
            
            # 1. Remove validação
            train_mask[val_start:val_end] = False
            
            # 2. Remove embargo ANTES da validação (purge)
            purge_start = max(0, val_start - self.purge_bars)
            train_mask[purge_start:val_start] = False
            
            # 3. Remove embargo DEPOIS da validação
            embargo_end = min(n_samples, val_end + self.embargo_bars)
            train_mask[val_end:embargo_end] = False
            
            train_indices = indices[train_mask]
            
            # Validação: não pode ter folds vazios
            if len(train_indices) == 0 or len(val_indices) == 0:
                logger.warning(
                    f"⚠️ [PurgedCV] Fold {fold_idx+1} vazio. "
                    f"Train={len(train_indices)}, Val={len(val_indices)}. Pulando."
                )
                continue
            
            logger.debug(
                f"[PurgedCV] Fold {fold_idx+1}/{self.n_splits}: "
                f"Train={len(train_indices)}, Val={len(val_indices)}"
            )
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Retorna número de splits (compatibilidade sklearn)."""
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    Melhoria sobre PurgedKFold:
    - Gera TODAS as combinações possíveis de N-1 paths como treino
    - Validação em 1 path
    - Mais robusto mas computacionalmente caro (2^N combinações)
    
    Referência:
        De Prado (2018), Section 7.4.2: "The Combinatorial Purged Cross-Validation"
    
    Nota: Complexidade cresce exponencialmente. Usar apenas para n_splits pequeno (≤6).
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_bars: int = 10,
        purge_bars: int = 0,
        max_combinations: int = 100
    ):
        """
        Args:
            n_splits: Número de paths temporais
            embargo_bars: Barras de embargo
            purge_bars: Barras de purging
            max_combinations: Limite de combinações (safety)
        """
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.purge_bars = purge_bars
        self.max_combinations = max_combinations
        
        # Calcula número de combinações
        from math import comb
        n_combos = comb(n_splits, n_splits - 1)
        
        if n_combos > max_combinations:
            logger.warning(
                f"⚠️ [CPCV] {n_combos} combinações excedem limite ({max_combinations}). "
                f"Considere reduzir n_splits ou usar PurgedKFold."
            )
        
        logger.info(f"📊 [CPCV] Configurado: {n_splits} splits, {n_combos} combinações")
    
    def split(
        self, 
        X: pd.DataFrame | np.ndarray,
        y: Optional[pd.Series | np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Gera todas as combinações de N-1 paths para treino.
        
        Yields:
            (train_indices, val_indices)
        """
        from itertools import combinations
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Divide em paths
        path_size = n_samples // self.n_splits
        paths = []
        for i in range(self.n_splits):
            start = i * path_size
            end = start + path_size if i < self.n_splits - 1 else n_samples
            paths.append((start, end))
        
        # Gera todas as combinações: escolhe 1 path para validação
        combo_count = 0
        for val_path_idx in range(self.n_splits):
            if combo_count >= self.max_combinations:
                logger.warning(f"⚠️ [CPCV] Limite de {self.max_combinations} atingido.")
                break
            
            val_start, val_end = paths[val_path_idx]
            val_indices = indices[val_start:val_end]
            
            # Treino: todos os outros paths (com embargo)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_start:val_end] = False
            
            # Purge antes
            purge_start = max(0, val_start - self.purge_bars)
            train_mask[purge_start:val_start] = False
            
            # Embargo depois
            embargo_end = min(n_samples, val_end + self.embargo_bars)
            train_mask[val_end:embargo_end] = False
            
            train_indices = indices[train_mask]
            
            if len(train_indices) == 0 or len(val_indices) == 0:
                continue
            
            combo_count += 1
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        from math import comb
        return min(comb(self.n_splits, self.n_splits - 1), self.max_combinations)


class TimeSeriesSplitWithEmbargo:
    """
    Variante do sklearn TimeSeriesSplit com embargo.
    
    Expansão incremental: cada fold adiciona mais dados ao conjunto de treino.
    
    Exemplo (n_splits=3, embargo=2):
        Fold 1: Train=[0,1,2], Embargo=[3,4], Val=[5,6]
        Fold 2: Train=[0,1,2,3,4,5,6], Embargo=[7,8], Val=[9,10]
        Fold 3: Train=[0...10], Embargo=[11,12], Val=[13,14]
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_bars: int = 10,
        min_train_size: Optional[int] = None
    ):
        """
        Args:
            n_splits: Número de folds
            embargo_bars: Barras de embargo
            min_train_size: Tamanho mínimo do conjunto de treino
        """
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.min_train_size = min_train_size
    
    def split(
        self, 
        X: pd.DataFrame | np.ndarray,
        y: Optional[pd.Series | np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Gera splits com treino crescente."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Tamanho de cada validação
        val_size = n_samples // (self.n_splits + 1)
        
        for fold_idx in range(self.n_splits):
            # Validação: bloco fixo
            val_start = (fold_idx + 1) * val_size + self.embargo_bars
            val_end = val_start + val_size
            
            if val_end > n_samples:
                break
            
            val_indices = indices[val_start:val_end]
            
            # Treino: tudo antes do embargo
            train_end = val_start - self.embargo_bars
            
            if self.min_train_size is not None:
                train_start = max(0, train_end - self.min_train_size)
            else:
                train_start = 0
            
            train_indices = indices[train_start:train_end]
            
            if len(train_indices) == 0 or len(val_indices) == 0:
                continue
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# Testes
if __name__ == "__main__":
    # Simula dados temporais
    n_samples = 100
    X = pd.DataFrame({'feature': np.arange(n_samples)})
    
    print("=== Purged K-Fold ===")
    pkf = PurgedKFold(n_splits=5, embargo_bars=5, purge_bars=3)
    
    for fold, (train_idx, val_idx) in enumerate(pkf.split(X)):
        print(f"Fold {fold+1}:")
        print(f"  Train: {len(train_idx)} samples, range [{train_idx.min()}, {train_idx.max()}]")
        print(f"  Val: {len(val_idx)} samples, range [{val_idx.min()}, {val_idx.max()}]")
        print(f"  Gap: {val_idx.min() - train_idx.max()}")
    
    print("\n=== Time Series Split with Embargo ===")
    tsse = TimeSeriesSplitWithEmbargo(n_splits=3, embargo_bars=5)
    
    for fold, (train_idx, val_idx) in enumerate(tsse.split(X)):
        print(f"Fold {fold+1}:")
        print(f"  Train: {len(train_idx)} samples, range [{train_idx.min()}, {train_idx.max()}]")
        print(f"  Val: {len(val_idx)} samples, range [{val_idx.min()}, {val_idx.max()}]")
