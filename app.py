import os
import threading
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from datetime import datetime

from utils.logger import get_logger
from config.settings import active_config
logger = get_logger("flask_app")
socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    """Cria e configura a instância da aplicação Flask."""
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config.from_object(active_config)
    
    socketio.init_app(app, async_mode='threading')
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/status', methods=['GET'])
    def get_status():
        """Retorna o estado geral e detalhado do sistema."""
        from run_bot import system_components, system_state
        try:
            portfolio_state = system_components["portfolio"].get_detailed_status()
            risk_profile = system_components["risk_manager"].get_current_risk_profile()
            ai_status = system_components["ai_controller"].get_status()
            
            full_status = {
                "bot_status": system_state["status"],
                "trading_active": system_state["trading_active"],
                "training_active": system_state["training_active"],
                "last_kline_update": system_state["last_kline_update"],
                "last_signal": system_state["last_signal"].__dict__ if system_state["last_signal"] else None,
                "portfolio": portfolio_state,
                "risk": risk_profile,
                "ai": ai_status
            }
            return jsonify(full_status)
        except Exception as e:
            logger.exception("Erro ao obter status do sistema.")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/control', methods=['POST'])
    def control_bot():
        """Controla o estado do bot (start/stop trading/training)."""
        from run_bot import system_components, system_state
        data = request.get_json()
        command = data.get('command')
        
        if command == 'start_trading':
            if not system_state["trading_active"]:
                logger.info("Comando da API para iniciar o trading recebido.")
                system_state["trading_active"] = True
                return jsonify({"status": "success", "message": "Trading ativado."})
            return jsonify({"status": "no_change", "message": "Trading já estava ativo."})
        
        elif command == 'stop_trading':
            if system_state["trading_active"]:
                logger.info("Comando da API para parar o trading recebido.")
                system_state["trading_active"] = False
                return jsonify({"status": "success", "message": "Trading desativado."})
            return jsonify({"status": "no_change", "message": "Trading já estava inativo."})
            
        elif command == 'start_training':
            if not system_state["training_active"]:
                logger.info("Comando da API para iniciar o treinamento recebido.")
                ai_controller = system_components.get("ai_controller")
                data_provider = system_components.get("data_provider")
                if ai_controller and data_provider:
                    ai_controller.start_training_mode(data_provider)
                    system_state["training_active"] = True
                    return jsonify({"status": "success", "message": "Treinamento em background iniciado."})
                else:
                    return jsonify({"status": "error", "message": "Componentes de IA ou Dados não inicializados."}), 500
            return jsonify({"status": "no_change", "message": "Treinamento já estava ativo."})

        elif command == 'stop_training':
            if system_state["training_active"]:
                logger.info("Comando da API para parar o treinamento recebido.")
                system_components["ai_controller"].stop_training_mode()
                system_state["training_active"] = False
                return jsonify({"status": "success", "message": "Treinamento parado."})
            return jsonify({"status": "no_change", "message": "Treinamento já estava inativo."})

        return jsonify({"status": "error", "message": "Comando inválido."}), 400

    @app.route('/api/explanation', methods=['GET'])
    def get_last_explanation():
        """Retorna a explicação da última decisão de trade."""
        from run_bot import system_state
        if system_state["last_signal"] and hasattr(system_state["last_signal"], 'explanation'):
            return jsonify(system_state["last_signal"].explanation)
        return jsonify({"message": "Nenhuma decisão recente para explicar."})

    def background_thread():
        """Envia atualizações de status para o frontend via SocketIO."""
        from run_bot import system_components, system_state, shutdown_event, training_status_queue
        while not shutdown_event.is_set():
            try:
                portfolio_state = system_components["portfolio"].get_detailed_status()
                status_data = {
                    "bot_status": system_state.get("status"),
                    "trading_active": system_state.get("trading_active"),
                    "training_active": system_state.get("training_active"),
                    "portfolio_value": portfolio_state.get('total_value', 0),
                    "pnl": portfolio_state.get('unrealized_pnl', 0),
                    "regime": system_components["ai_controller"].current_market_regime,
                }
                socketio.emit('status_update', status_data)

                while not training_status_queue.empty():
                    log_msg = training_status_queue.get()
                    socketio.emit('training_log', {'log': log_msg})
                    training_status_queue.task_done()

                socketio.sleep(2)
            except Exception as e:
                logger.error(f"Erro no broadcast do SocketIO: {e}")
                socketio.sleep(15)

    @socketio.on('connect')
    def on_connect():
        logger.info('Cliente conectado ao WebSocket.')
        if not getattr(app, 'background_thread_started', False):
            threading.Thread(target=background_thread, daemon=True).start()
            app.background_thread_started = True
            logger.info('Thread de broadcast do SocketIO iniciada.')

    return app