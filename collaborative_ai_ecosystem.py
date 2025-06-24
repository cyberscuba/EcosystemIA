import asyncio
import json
import time
import uuid
import platform
import psutil
import sqlite3
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import websockets
import requests
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from jinja2 import Template

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class AgentType(Enum):
    MONITOR = "monitor"
    HEALER = "healer"
    COORDINATOR = "coordinator"
    REPORTER = "reporter"
    SECURITY = "security"

class MessageType(Enum):
    METRICS = "metrics"
    ALERT = "alert"
    HEALING_REQUEST = "healing_request"
    HEALING_RESPONSE = "healing_response"
    COORDINATION = "coordination"
    REPORT_REQUEST = "report_request"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentMessage:
    id: str
    sender_id: str
    sender_type: AgentType
    message_type: MessageType
    timestamp: str
    data: Dict[str, Any]
    target_agents: Optional[List[str]] = None
    correlation_id: Optional[str] = None

@dataclass
class SystemMetrics:
    agent_id: str
    hostname: str
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    processes_count: int
    load_average: float
    service_status: Dict[str, bool]

@dataclass
class HealingAction:
    id: str
    target_system: str
    problem_type: str
    action_type: str
    commands: List[str]
    expected_outcome: str
    rollback_commands: List[str]
    timeout_seconds: int
    priority: Severity

@dataclass
class ExecutiveReport:
    id: str
    timestamp: str
    period: str
    systems_summary: Dict[str, Any]
    incidents_summary: Dict[str, Any]
    healing_summary: Dict[str, Any]
    performance_trends: Dict[str, Any]
    recommendations: List[str]
    charts: Dict[str, str]  # chart_name -> base64_encoded_image

class BaseAgent:
    """Clase base para todos los agentes colaborativos"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.logger = logging.getLogger(f"{agent_type.value}_{agent_id}")
        self.is_running = False
        self.message_queue = asyncio.Queue()
        self.peer_agents = {}  # agent_id -> websocket
        self.db_path = f"agent_{agent_id}.db"
        
        # WebSocket server para comunicación entre agentes
        self.ws_port = config.get('ws_port', 8000 + hash(agent_id) % 1000)
        
        self._init_database()
        self.logger.info(f"Agente {agent_type.value} inicializado en puerto {self.ws_port}")
    
    def _init_database(self):
        """Inicializar base de datos del agente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                sender_id TEXT,
                message_type TEXT,
                timestamp TEXT,
                data TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS actions (
                id TEXT PRIMARY KEY,
                action_type TEXT,
                target_system TEXT,
                commands TEXT,
                timestamp TEXT,
                status TEXT,
                result TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def send_message(self, message: AgentMessage):
        """Enviar mensaje a otros agentes"""
        try:
            # Guardar mensaje en DB
            self._save_message(message)
            
            # Enviar a agentes específicos o broadcast
            targets = message.target_agents or list(self.peer_agents.keys())
            
            for target_id in targets:
                if target_id in self.peer_agents:
                    try:
                        await self.peer_agents[target_id].send(json.dumps(asdict(message)))
                        self.logger.debug(f"Mensaje enviado a {target_id}")
                    except Exception as e:
                        self.logger.error(f"Error enviando a {target_id}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error enviando mensaje: {e}")
    
    def _save_message(self, message: AgentMessage):
        """Guardar mensaje en base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (id, sender_id, message_type, timestamp, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (message.id, message.sender_id, message.message_type.value, 
              message.timestamp, json.dumps(message.data)))
        
        conn.commit()
        conn.close()
    
    async def handle_message(self, message: AgentMessage):
        """Manejar mensaje recibido - implementar en subclases"""
        pass
    
    async def start_websocket_server(self):
        """Iniciar servidor WebSocket para comunicación"""
        async def handle_client(websocket, path):
            try:
                async for message_raw in websocket:
                    try:
                        message_dict = json.loads(message_raw)
                        message = AgentMessage(**message_dict)
                        await self.message_queue.put(message)
                        self.logger.debug(f"Mensaje recibido de {message.sender_id}")
                    except Exception as e:
                        self.logger.error(f"Error procesando mensaje: {e}")
            except websockets.exceptions.ConnectionClosed:
                self.logger.info("Cliente desconectado")
        
        start_server = websockets.serve(handle_client, "localhost", self.ws_port)
        await start_server
        self.logger.info(f"Servidor WebSocket iniciado en puerto {self.ws_port}")
    
    async def connect_to_peer(self, peer_id: str, peer_port: int):
        """Conectar a otro agente"""
        try:
            uri = f"ws://localhost:{peer_port}"
            websocket = await websockets.connect(uri)
            self.peer_agents[peer_id] = websocket
            self.logger.info(f"Conectado a agente {peer_id}")
        except Exception as e:
            self.logger.error(f"Error conectando a {peer_id}: {e}")
    
    async def run(self):
        """Ejecutar agente principal"""
        self.is_running = True
        
        # Iniciar servidor WebSocket
        await self.start_websocket_server()
        
        # Procesar mensajes
        while self.is_running:
            try:
                # Procesar mensajes de la cola
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                    await self.handle_message(message)
                except asyncio.TimeoutError:
                    pass
                
                # Ejecutar lógica específica del agente
                await self.agent_logic()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error en bucle principal: {e}")
                await asyncio.sleep(1)
    
    async def agent_logic(self):
        """Lógica específica del agente - implementar en subclases"""
        pass

class MonitorAgent(BaseAgent):
    """Agente especializado en monitoreo de sistemas"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.MONITOR, config)
        self.last_metrics_time = 0
        self.metrics_interval = config.get('metrics_interval', 30)
        self.alert_thresholds = config.get('thresholds', {
            'cpu': 80,
            'memory': 85,
            'disk': 90
        })
    
    async def collect_metrics(self) -> SystemMetrics:
        """Recopilar métricas del sistema"""
        # Métricas básicas
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Promedio de uso de disco
        disk_usage = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage.append((usage.used / usage.total) * 100)
            except:
                continue
        disk_percent = np.mean(disk_usage) if disk_usage else 0
        
        # Red
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv
        }
        
        # Load average
        try:
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else cpu_percent / 100
        except:
            load_avg = cpu_percent / 100
        
        # Estado de servicios críticos
        service_status = await self._check_services()
        
        return SystemMetrics(
            agent_id=self.agent_id,
            hostname=platform.node(),
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk_percent,
            network_io=network_io,
            processes_count=len(psutil.pids()),
            load_average=load_avg,
            service_status=service_status
        )
    
    async def _check_services(self) -> Dict[str, bool]:
        """Verificar estado de servicios críticos"""
        services = self.config.get('critical_services', ['ssh', 'nginx', 'postgresql'])
        status = {}
        
        for service in services:
            try:
                if platform.system() == "Linux":
                    result = subprocess.run(['systemctl', 'is-active', service], 
                                          capture_output=True, text=True, timeout=5)
                    status[service] = result.stdout.strip() == 'active'
                else:
                    # Windows services check
                    result = subprocess.run(['sc', 'query', service], 
                                          capture_output=True, text=True, timeout=5)
                    status[service] = 'RUNNING' in result.stdout
            except:
                status[service] = False
        
        return status
    
    async def check_alerts(self, metrics: SystemMetrics):
        """Verificar si hay alertas que enviar"""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > self.alert_thresholds['cpu']:
            alerts.append({
                'type': 'cpu_high',
                'severity': Severity.CRITICAL if metrics.cpu_percent > 95 else Severity.HIGH,
                'value': metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu'],
                'message': f"Alto uso de CPU: {metrics.cpu_percent}%"
            })
        
        # Memory alert
        if metrics.memory_percent > self.alert_thresholds['memory']:
            alerts.append({
                'type': 'memory_high',
                'severity': Severity.CRITICAL if metrics.memory_percent > 95 else Severity.HIGH,
                'value': metrics.memory_percent,
                'threshold': self.alert_thresholds['memory'],
                'message': f"Alto uso de memoria: {metrics.memory_percent}%"
            })
        
        # Disk alert
        if metrics.disk_percent > self.alert_thresholds['disk']:
            alerts.append({
                'type': 'disk_high',
                'severity': Severity.CRITICAL if metrics.disk_percent > 95 else Severity.HIGH,
                'value': metrics.disk_percent,
                'threshold': self.alert_thresholds['disk'],
                'message': f"Alto uso de disco: {metrics.disk_percent}%"
            })
        
        # Service alerts
        for service, is_running in metrics.service_status.items():
            if not is_running:
                alerts.append({
                    'type': 'service_down',
                    'severity': Severity.CRITICAL,
                    'service': service,
                    'message': f"Servicio {service} no está ejecutándose"
                })
        
        # Enviar alertas a agentes de healing
        for alert in alerts:
            await self.send_alert(alert, metrics)
    
    async def send_alert(self, alert: Dict[str, Any], metrics: SystemMetrics):
        """Enviar alerta a agentes de healing"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            sender_type=self.agent_type,
            message_type=MessageType.ALERT,
            timestamp=datetime.now().isoformat(),
            data={
                'alert': alert,
                'metrics': asdict(metrics),
                'hostname': platform.node()
            }
        )
        
        await self.send_message(message)
        self.logger.warning(f"ALERTA: {alert['message']}")
    
    async def agent_logic(self):
        """Lógica principal del agente monitor"""
        current_time = time.time()
        
        if current_time - self.last_metrics_time >= self.metrics_interval:
            try:
                # Recopilar métricas
                metrics = await self.collect_metrics()
                
                # Enviar métricas a coordinador
                metrics_message = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    sender_type=self.agent_type,
                    message_type=MessageType.METRICS,
                    timestamp=datetime.now().isoformat(),
                    data=asdict(metrics)
                )
                
                await self.send_message(metrics_message)
                
                # Verificar alertas
                await self.check_alerts(metrics)
                
                self.last_metrics_time = current_time
                self.logger.info(f"Métricas enviadas - CPU: {metrics.cpu_percent}%, RAM: {metrics.memory_percent}%")
                
            except Exception as e:
                self.logger.error(f"Error recopilando métricas: {e}")

class HealerAgent(BaseAgent):
    """Agente especializado en auto-healing"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.HEALER, config)
        self.healing_actions = self._load_healing_playbook()
        self.active_healings = {}  # healing_id -> HealingAction
        self.max_concurrent_healings = config.get('max_concurrent_healings', 3)
    
    def _load_healing_playbook(self) -> Dict[str, HealingAction]:
        """Cargar playbook de acciones de healing"""
        playbook = {
            'cpu_high': HealingAction(
                id='cpu_high_standard',
                target_system='local',
                problem_type='cpu_high',
                action_type='process_optimization',
                commands=[
                    'ps aux --sort=-%cpu | head -10',
                    'top -b -n1 -o %CPU | head -15',
                    'kill -STOP $(ps aux --sort=-%cpu | awk "NR==2{print $2}")',  # Parar proceso más intensivo
                    'sleep 30',
                    'kill -CONT $(ps aux --sort=-%cpu | awk "NR==2{print $2}")'  # Reanudar proceso
                ],
                expected_outcome='CPU usage below 70%',
                rollback_commands=[
                    'kill -CONT $(ps aux --sort=-%cpu | awk "NR==2{print $2}")'
                ],
                timeout_seconds=300,
                priority=Severity.HIGH
            ),
            
            'memory_high': HealingAction(
                id='memory_high_standard',
                target_system='local',
                problem_type='memory_high',
                action_type='memory_cleanup',
                commands=[
                    'sync',
                    'echo 1 > /proc/sys/vm/drop_caches',
                    'systemctl restart tmpfs-cleaner || true',
                    'find /tmp -type f -mtime +1 -delete || true'
                ],
                expected_outcome='Memory usage below 80%',
                rollback_commands=[],
                timeout_seconds=120,
                priority=Severity.HIGH
            ),
            
            'disk_high': HealingAction(
                id='disk_high_standard',
                target_system='local',
                problem_type='disk_high',
                action_type='disk_cleanup',
                commands=[
                    'journalctl --vacuum-time=7d',
                    'apt-get autoremove -y || yum autoremove -y || true',
                    'find /var/log -name "*.log" -mtime +7 -exec gzip {} \\;',
                    'docker system prune -f || true'
                ],
                expected_outcome='Disk usage below 85%',
                rollback_commands=[],
                timeout_seconds=600,
                priority=Severity.MEDIUM
            ),
            
            'service_down': HealingAction(
                id='service_restart',
                target_system='local',
                problem_type='service_down',
                action_type='service_restart',
                commands=[
                    'systemctl restart {service_name}',
                    'sleep 10',
                    'systemctl status {service_name}'
                ],
                expected_outcome='Service running and active',
                rollback_commands=[
                    'systemctl stop {service_name}'
                ],
                timeout_seconds=60,
                priority=Severity.CRITICAL
            )
        }
        
        return playbook
    
    async def handle_message(self, message: AgentMessage):
        """Manejar mensajes de otros agentes"""
        if message.message_type == MessageType.ALERT:
            await self.handle_alert(message)
    
    async def handle_alert(self, message: AgentMessage):
        """Manejar alerta y ejecutar healing si es necesario"""
        try:
            alert_data = message.data['alert']
            alert_type = alert_data['type']
            severity = alert_data['severity']
            
            self.logger.info(f"Recibida alerta: {alert_type} - {severity}")
            
            # Verificar si tenemos acción de healing para este tipo de alerta
            if alert_type in self.healing_actions:
                # Verificar límite de healings concurrentes
                if len(self.active_healings) < self.max_concurrent_healings:
                    await self.execute_healing(alert_type, alert_data, message.data)
                else:
                    self.logger.warning(f"Máximo de healings concurrentes alcanzado. Alerta {alert_type} en cola.")
            else:
                self.logger.warning(f"No hay acción de healing definida para {alert_type}")
                
        except Exception as e:
            self.logger.error(f"Error manejando alerta: {e}")
    
    async def execute_healing(self, alert_type: str, alert_data: Dict[str, Any], context: Dict[str, Any]):
        """Ejecutar acción de healing"""
        healing_action = self.healing_actions[alert_type].copy()
        healing_id = str(uuid.uuid4())
        
        # Personalizar comandos si es necesario
        if alert_type == 'service_down':
            service_name = alert_data.get('service', 'unknown')
            healing_action.commands = [cmd.format(service_name=service_name) for cmd in healing_action.commands]
        
        self.active_healings[healing_id] = healing_action
        
        self.logger.info(f"Iniciando healing {healing_id} para {alert_type}")
        
        try:
            # Ejecutar comandos de healing
            success = await self._execute_commands(healing_action.commands, healing_action.timeout_seconds)
            
            if success:
                # Verificar si el healing fue exitoso
                verification_result = await self._verify_healing_success(alert_type, alert_data)
                
                if verification_result:
                    self.logger.info(f"Healing {healing_id} completado exitosamente")
                    await self._notify_healing_success(healing_id, alert_type, context)
                else:
                    self.logger.warning(f"Healing {healing_id} ejecutado pero problema persiste")
                    await self._execute_rollback(healing_action)
            else:
                self.logger.error(f"Healing {healing_id} falló")
                await self._execute_rollback(healing_action)
                
        except Exception as e:
            self.logger.error(f"Error ejecutando healing {healing_id}: {e}")
            await self._execute_rollback(healing_action)
        
        finally:
            # Limpiar healing activo
            if healing_id in self.active_healings:
                del self.active_healings[healing_id]
    
    async def _execute_commands(self, commands: List[str], timeout: int) -> bool:
        """Ejecutar lista de comandos con timeout"""
        try:
            for command in commands:
                self.logger.info(f"Ejecutando: {command}")
                
                # Ejecutar comando de forma segura
                if self._is_safe_command(command):
                    result = subprocess.run(
                        command.split(),
                        capture_output=True,
                        text=True,
                        timeout=timeout // len(commands)
                    )
                    
                    if result.returncode != 0 and 'true' not in command:
                        self.logger.error(f"Comando falló: {command} - {result.stderr}")
                        return False
                    
                    self.logger.debug(f"Comando exitoso: {result.stdout[:200]}")
                else:
                    self.logger.error(f"Comando no seguro rechazado: {command}")
                    return False
            
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout ejecutando comandos de healing")
            return False
        except Exception as e:
            self.logger.error(f"Error ejecutando comandos: {e}")
            return False
    
    def _is_safe_command(self, command: str) -> bool:
        """Verificar que el comando sea seguro de ejecutar"""
        dangerous_patterns = [
            'rm -rf /', 'format', 'mkfs', 'dd if=', '> /dev/', 'chmod 777',
            'passwd', 'userdel', 'groupdel', 'crontab -r'
        ]
        
        dangerous_commands = [
            'shutdown', 'reboot', 'halt', 'poweroff', 'init 0', 'init 6'
        ]
        
        cmd_lower = command.lower()
        
        # Verificar patrones peligrosos
        for pattern in dangerous_patterns:
            if pattern in cmd_lower:
                return False
        
        # Verificar comandos peligrosos
        for dangerous_cmd in dangerous_commands:
            if cmd_lower.startswith(dangerous_cmd):
                return False
        
        return True
    
    async def _verify_healing_success(self, alert_type: str, alert_data: Dict[str, Any]) -> bool:
        """Verificar si el healing fue exitoso"""
        try:
            if alert_type == 'cpu_high':
                current_cpu = psutil.cpu_percent(interval=2)
                return current_cpu < alert_data['threshold']
            
            elif alert_type == 'memory_high':
                current_memory = psutil.virtual_memory().percent
                return current_memory < alert_data['threshold']
            
            elif alert_type == 'disk_high':
                # Verificar uso de disco promedio
                disk_usages = []
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disk_usages.append((usage.used / usage.total) * 100)
                    except:
                        continue
                current_disk = np.mean(disk_usages) if disk_usages else 0
                return current_disk < alert_data['threshold']
            
            elif alert_type == 'service_down':
                service_name = alert_data.get('service')
                if service_name:
                    result = subprocess.run(['systemctl', 'is-active', service_name], 
                                          capture_output=True, text=True, timeout=5)
                    return result.stdout.strip() == 'active'
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error verificando healing: {e}")
            return False
    
    async def _execute_rollback(self, healing_action: HealingAction):
        """Ejecutar comandos de rollback"""
        if healing_action.rollback_commands:
            self.logger.info("Ejecutando rollback...")
            await self._execute_commands(healing_action.rollback_commands, 60)
    
    async def _notify_healing_success(self, healing_id: str, alert_type: str, context: Dict[str, Any]):
        """Notificar éxito del healing"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            sender_type=self.agent_type,
            message_type=MessageType.HEALING_RESPONSE,
            timestamp=datetime.now().isoformat(),
            data={
                'healing_id': healing_id,
                'alert_type': alert_type,
                'status': 'success',
                'context': context
            }
        )
        
        await self.send_message(message)
    
    async def agent_logic(self):
        """Lógica principal del agente healer"""
        # Monitorear healings activos
        for healing_id, action in list(self.active_healings.items()):
            # Aquí podrías implementar monitoreo de progreso de healings
            pass

class ReporterAgent(BaseAgent):
    """Agente especializado en generar reportes ejecutivos"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.REPORTER, config)
        self.collected_metrics = []
        self.incidents_log = []
        self.healing_log = []
        self.report_interval = config.get('report_interval', 3600)  # 1 hora
        self.last_report_time = 0
    
    async def handle_message(self, message: AgentMessage):
        """Manejar mensajes de otros agentes"""
        if message.message_type == MessageType.METRICS:
            self.collected_metrics.append(message.data)
            # Mantener solo últimas 24 horas de métricas
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.collected_metrics = [
                m for m in self.collected_metrics 
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
        
        elif message.message_type == MessageType.ALERT:
            self.incidents_log.append({
                'timestamp': message.timestamp,
                'alert': message.data['alert'],
                'hostname': message.data.get('hostname', 'unknown')
            })
        
        elif message.message_type == MessageType.HEALING_RESPONSE:
            self.healing_log.append(message.data)
    
    async def generate_executive_report(self) -> ExecutiveReport:
        """Generar reporte ejecutivo completo"""
        try:
            report_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Analizar métricas
            systems_summary = self._analyze_systems_performance()
            incidents_summary = self._analyze_incidents()
            healing_summary = self._analyze_healing_effectiveness()
            performance_trends = self._analyze_performance_trends()
            recommendations = self._generate_recommendations()
            
            # Generar gráficos
            charts = await self._generate_charts()
            
            report = ExecutiveReport(
                id=report_id,
                timestamp=timestamp,
                period="Últimas 24 horas",
                systems_summary=systems_summary,
                incidents_summary=incidents_summary,
                healing_summary=healing_summary,
                performance_trends=performance_trends,
                recommendations=recommendations,
                charts=charts
            )
            
            # Guardar reporte
            await self._save_report(report)
            
            # Generar HTML
            await self._generate_html_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            raise
    
    def _analyze_systems_performance(self) -> Dict[str, Any]:
        """Analizar rendimiento de sistemas"""
        if not self.collected_metrics:
            return {"status": "No hay datos disponibles"}
        
        # Calcular estadísticas
        summary = {}
        total_systems = len(systems)
        healthy_systems = 0
        
        for hostname, data in systems.items():
            avg_cpu = np.mean(data['cpu_values'])
            avg_memory = np.mean(data['memory_values'])
            avg_disk = np.mean(data['disk_values'])
            max_cpu = np.max(data['cpu_values'])
            max_memory = np.max(data['memory_values'])
            
            # Determinar estado del sistema
            status = "HEALTHY"
            if max_cpu > 90 or max_memory > 90:
                status = "CRITICAL"
            elif avg_cpu > 70 or avg_memory > 75:
                status = "WARNING"
            else:
                healthy_systems += 1
            
            summary[hostname] = {
                'status': status,
                'avg_cpu': round(avg_cpu, 2),
                'avg_memory': round(avg_memory, 2),
                'avg_disk': round(avg_disk, 2),
                'max_cpu': round(max_cpu, 2),
                'max_memory': round(max_memory, 2),
                'last_seen': data['last_seen'],
                'uptime_percentage': 100.0  # Simplificado
            }
        
        return {
            'total_systems': total_systems,
            'healthy_systems': healthy_systems,
            'systems_detail': summary,
            'overall_health': round((healthy_systems / total_systems) * 100, 1) if total_systems > 0 else 0
        }
    
    def _analyze_incidents(self) -> Dict[str, Any]:
        """Analizar incidentes del período"""
        # Filtrar últimas 24 horas
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_incidents = [
            inc for inc in self.incidents_log
            if datetime.fromisoformat(inc['timestamp']) > cutoff_time
        ]
        
        if not recent_incidents:
            return {
                'total_incidents': 0,
                'by_severity': {},
                'by_type': {},
                'by_system': {},
                'resolution_rate': 100.0
            }
        
        # Análisis por severidad
        by_severity = {}
        by_type = {}
        by_system = {}
        
        for incident in recent_incidents:
            alert = incident['alert']
            severity = alert.get('severity', 'unknown')
            alert_type = alert.get('type', 'unknown')
            hostname = incident.get('hostname', 'unknown')
            
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
            by_system[hostname] = by_system.get(hostname, 0) + 1
        
        # Calcular tasa de resolución (simplificado)
        resolved_incidents = len([h for h in self.healing_log if h.get('status') == 'success'])
        resolution_rate = (resolved_incidents / len(recent_incidents)) * 100 if recent_incidents else 100
        
        return {
            'total_incidents': len(recent_incidents),
            'by_severity': by_severity,
            'by_type': by_type,
            'by_system': by_system,
            'resolution_rate': round(resolution_rate, 1)
        }
    
    def _analyze_healing_effectiveness(self) -> Dict[str, Any]:
        """Analizar efectividad del auto-healing"""
        if not self.healing_log:
            return {
                'total_healings': 0,
                'success_rate': 0,
                'by_type': {},
                'avg_resolution_time': 0
            }
        
        # Filtrar últimas 24 horas
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_healings = [
            h for h in self.healing_log
            if datetime.fromisoformat(h.get('timestamp', datetime.now().isoformat())) > cutoff_time
        ]
        
        successful_healings = [h for h in recent_healings if h.get('status') == 'success']
        
        # Análisis por tipo
        by_type = {}
        for healing in recent_healings:
            alert_type = healing.get('alert_type', 'unknown')
            status = healing.get('status', 'unknown')
            
            if alert_type not in by_type:
                by_type[alert_type] = {'total': 0, 'successful': 0}
            
            by_type[alert_type]['total'] += 1
            if status == 'success':
                by_type[alert_type]['successful'] += 1
        
        # Calcular tasas de éxito por tipo
        for alert_type in by_type:
            total = by_type[alert_type]['total']
            successful = by_type[alert_type]['successful']
            by_type[alert_type]['success_rate'] = round((successful / total) * 100, 1) if total > 0 else 0
        
        success_rate = round((len(successful_healings) / len(recent_healings)) * 100, 1) if recent_healings else 0
        
        return {
            'total_healings': len(recent_healings),
            'successful_healings': len(successful_healings),
            'success_rate': success_rate,
            'by_type': by_type,
            'avg_resolution_time': 45  # Simulado - en segundos
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analizar tendencias de rendimiento"""
        if len(self.collected_metrics) < 10:
            return {"status": "Datos insuficientes para análisis de tendencias"}
        
        # Ordenar métricas por tiempo
        sorted_metrics = sorted(self.collected_metrics, key=lambda x: x['timestamp'])
        
        # Dividir en dos mitades para comparar tendencias
        mid_point = len(sorted_metrics) // 2
        first_half = sorted_metrics[:mid_point]
        second_half = sorted_metrics[mid_point:]
        
        def calculate_averages(metrics_list):
            return {
                'cpu': np.mean([m['cpu_percent'] for m in metrics_list]),
                'memory': np.mean([m['memory_percent'] for m in metrics_list]),
                'disk': np.mean([m['disk_percent'] for m in metrics_list])
            }
        
        first_avg = calculate_averages(first_half)
        second_avg = calculate_averages(second_half)
        
        trends = {}
        for metric in ['cpu', 'memory', 'disk']:
            change = second_avg[metric] - first_avg[metric]
            if abs(change) < 2:
                trend = "estable"
            elif change > 0:
                trend = "creciente"
            else:
                trend = "decreciente"
            
            trends[metric] = {
                'trend': trend,
                'change_percentage': round(change, 2),
                'current_avg': round(second_avg[metric], 2),
                'previous_avg': round(first_avg[metric], 2)
            }
        
        return trends
    
    def _generate_recommendations(self) -> List[str]:
        """Generar recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Analizar métricas recientes
        if self.collected_metrics:
            latest_metrics = self.collected_metrics[-10:] if len(self.collected_metrics) >= 10 else self.collected_metrics
            
            avg_cpu = np.mean([m['cpu_percent'] for m in latest_metrics])
            avg_memory = np.mean([m['memory_percent'] for m in latest_metrics])
            avg_disk = np.mean([m['disk_percent'] for m in latest_metrics])
            
            if avg_cpu > 70:
                recommendations.append("🔥 Considerar optimización de procesos con alto uso de CPU o escalamiento horizontal")
            
            if avg_memory > 75:
                recommendations.append("🧠 Evaluar implementación de cache más eficiente o incremento de memoria RAM")
            
            if avg_disk > 80:
                recommendations.append("💾 Implementar rotación de logs y limpieza automática de archivos temporales")
        
        # Analizar incidentes
        recent_incidents = [
            inc for inc in self.incidents_log
            if datetime.fromisoformat(inc['timestamp']) > datetime.now() - timedelta(hours=24)
        ]
        
        if len(recent_incidents) > 5:
            recommendations.append("⚠️ Alto número de incidentes. Revisar umbrales de alertas y estabilidad del sistema")
        
        # Analizar efectividad de healing
        if self.healing_log:
            success_rate = len([h for h in self.healing_log if h.get('status') == 'success']) / len(self.healing_log) * 100
            if success_rate < 80:
                recommendations.append("🔧 Revisar y optimizar scripts de auto-healing para mejorar tasa de éxito")
        
        # Recomendaciones generales
        if not recommendations:
            recommendations.append("✅ Sistema funcionando óptimamente. Continuar con monitoreo preventivo")
        
        recommendations.append("📊 Implementar dashboards en tiempo real para visibilidad mejorada")
        recommendations.append("🔄 Configurar backups automáticos y pruebas de recuperación")
        
        return recommendations[:5]  # Máximo 5 recomendaciones
    
    async def _generate_charts(self) -> Dict[str, str]:
        """Generar gráficos para el reporte"""
        charts = {}
        
        if not self.collected_metrics:
            return charts
        
        try:
            # Preparar datos
            df = pd.DataFrame(self.collected_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Gráfico 1: Tendencias de CPU y Memoria
            fig1 = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Uso de CPU (%)', 'Uso de Memoria (%)'),
                vertical_spacing=0.1
            )
            
            fig1.add_trace(
                go.Scatter(x=df['timestamp'], y=df['cpu_percent'], 
                          name='CPU', line=dict(color='#ff6b6b')),
                row=1, col=1
            )
            
            fig1.add_trace(
                go.Scatter(x=df['timestamp'], y=df['memory_percent'], 
                          name='Memoria', line=dict(color='#4ecdc4')),
                row=2, col=1
            )
            
            fig1.update_layout(
                title="Tendencias de Rendimiento - Últimas 24h",
                height=600,
                showlegend=False
            )
            
            charts['performance_trends'] = fig1.to_html(include_plotlyjs='cdn')
            
            # Gráfico 2: Distribución de Incidentes
            if self.incidents_log:
                incident_types = [inc['alert']['type'] for inc in self.incidents_log]
                incident_counts = pd.Series(incident_types).value_counts()
                
                fig2 = go.Figure(data=[
                    go.Pie(labels=incident_counts.index, values=incident_counts.values,
                           hole=0.4, marker_colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
                ])
                
                fig2.update_layout(
                    title="Distribución de Incidentes por Tipo",
                    height=400
                )
                
                charts['incidents_distribution'] = fig2.to_html(include_plotlyjs='cdn')
            
            # Gráfico 3: Efectividad de Healing
            if self.healing_log:
                healing_success = [h.get('status', 'unknown') for h in self.healing_log]
                success_counts = pd.Series(healing_success).value_counts()
                
                fig3 = go.Figure(data=[
                    go.Bar(x=success_counts.index, y=success_counts.values,
                           marker_color=['#28a745' if x == 'success' else '#dc3545' for x in success_counts.index])
                ])
                
                fig3.update_layout(
                    title="Efectividad del Auto-Healing",
                    xaxis_title="Estado",
                    yaxis_title="Cantidad",
                    height=400
                )
                
                charts['healing_effectiveness'] = fig3.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Error generando gráficos: {e}")
        
        return charts
    
    async def _save_report(self, report: ExecutiveReport):
        """Guardar reporte en base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                period TEXT,
                data TEXT
            )
        ''')
        
        cursor.execute('''
            INSERT INTO reports (id, timestamp, period, data)
            VALUES (?, ?, ?, ?)
        ''', (report.id, report.timestamp, report.period, json.dumps(asdict(report))))
        
        conn.commit()
        conn.close()
    
    async def _generate_html_report(self, report: ExecutiveReport):
        """Generar reporte HTML ejecutivo"""
        template_html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Ejecutivo de Sistemas</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; color: #333; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 5px solid #667eea; }
        .metric-card h3 { color: #667eea; margin-bottom: 15px; font-size: 1.1em; }
        .metric-value { font-size: 2.5em; font-weight: bold; color: #333; margin-bottom: 5px; }
        .metric-label { color: #666; font-size: 0.9em; }
        .section { background: white; margin-bottom: 30px; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .section-header { background: #667eea; color: white; padding: 20px; font-size: 1.3em; font-weight: bold; }
        .section-content { padding: 25px; }
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .recommendations { list-style: none; }
        .recommendations li { padding: 10px 0; border-bottom: 1px solid #eee; font-size: 1.1em; }
        .recommendations li:last-child { border-bottom: none; }
        .chart-container { margin: 20px 0; }
        .systems-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .systems-table th, .systems-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .systems-table th { background: #f8f9fa; font-weight: bold; }
        .trend-up { color: #dc3545; }
        .trend-down { color: #28a745; }
        .trend-stable { color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Reporte Ejecutivo de Sistemas</h1>
            <p>{{ report.period }} | Generado: {{ report.timestamp[:19] }}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>📊 Sistemas Totales</h3>
                <div class="metric-value">{{ report.systems_summary.total_systems }}</div>
                <div class="metric-label">Sistemas monitoreados</div>
            </div>
            
            <div class="metric-card">
                <h3>✅ Salud General</h3>
                <div class="metric-value {{ 'status-healthy' if report.systems_summary.overall_health > 80 else 'status-warning' if report.systems_summary.overall_health > 60 else 'status-critical' }}">
                    {{ report.systems_summary.overall_health }}%
                </div>
                <div class="metric-label">Sistemas saludables</div>
            </div>
            
            <div class="metric-card">
                <h3>🚨 Incidentes</h3>
                <div class="metric-value {{ 'status-healthy' if report.incidents_summary.total_incidents < 5 else 'status-warning' if report.incidents_summary.total_incidents < 15 else 'status-critical' }}">
                    {{ report.incidents_summary.total_incidents }}
                </div>
                <div class="metric-label">Últimas 24 horas</div>
            </div>
            
            <div class="metric-card">
                <h3>🔧 Auto-Healing</h3>
                <div class="metric-value {{ 'status-healthy' if report.healing_summary.success_rate > 80 else 'status-warning' if report.healing_summary.success_rate > 60 else 'status-critical' }}">
                    {{ report.healing_summary.success_rate }}%
                </div>
                <div class="metric-label">Tasa de éxito</div>
            </div>
        </div>
        
        {% if report.charts.performance_trends %}
        <div class="section">
            <div class="section-header">📈 Tendencias de Rendimiento</div>
            <div class="section-content">
                <div class="chart-container">
                    {{ report.charts.performance_trends|safe }}
                </div>
            </div>
        </div>
        {% endif %}
        
        <div class="section">
            <div class="section-header">🖥️ Estado de Sistemas</div>
            <div class="section-content">
                <table class="systems-table">
                    <thead>
                        <tr>
                            <th>Sistema</th>
                            <th>Estado</th>
                            <th>CPU Promedio</th>
                            <th>Memoria Promedio</th>
                            <th>Disco Promedio</th>
                            <th>Última Conexión</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for hostname, data in report.systems_summary.systems_detail.items() %}
                        <tr>
                            <td><strong>{{ hostname }}</strong></td>
                            <td><span class="{{ 'status-healthy' if data.status == 'HEALTHY' else 'status-warning' if data.status == 'WARNING' else 'status-critical' }}">{{ data.status }}</span></td>
                            <td>{{ data.avg_cpu }}%</td>
                            <td>{{ data.avg_memory }}%</td>
                            <td>{{ data.avg_disk }}%</td>
                            <td>{{ data.last_seen[:19] }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        {% if report.performance_trends.status is not defined %}
        <div class="section">
            <div class="section-header">📊 Análisis de Tendencias</div>
            <div class="section-content">
                <div class="metrics-grid">
                    {% for metric, data in report.performance_trends.items() %}
                    <div class="metric-card">
                        <h3>{{ metric.upper() }}</h3>
                        <div class="metric-value {{ 'trend-up' if data.trend == 'creciente' else 'trend-down' if data.trend == 'decreciente' else 'trend-stable' }}">
                            {{ data.trend }}
                        </div>
                        <div class="metric-label">{{ data.change_percentage }}% cambio | Actual: {{ data.current_avg }}%</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endif %}
        
        {% if report.charts.incidents_distribution %}
        <div class="section">
            <div class="section-header">🔍 Análisis de Incidentes</div>
            <div class="section-content">
                <div class="chart-container">
                    {{ report.charts.incidents_distribution|safe }}
                </div>
                <p><strong>Tasa de resolución:</strong> {{ report.incidents_summary.resolution_rate }}%</p>
            </div>
        </div>
        {% endif %}
        
        {% if report.charts.healing_effectiveness %}
        <div class="section">
            <div class="section-header">🤖 Efectividad del Auto-Healing</div>
            <div class="section-content">
                <div class="chart-container">
                    {{ report.charts.healing_effectiveness|safe }}
                </div>
                <p><strong>Total de intervenciones:</strong> {{ report.healing_summary.total_healings }}</p>
                <p><strong>Intervenciones exitosas:</strong> {{ report.healing_summary.successful_healings }}</p>
            </div>
        </div>
        {% endif %}
        
        <div class="section">
            <div class="section-header">💡 Recomendaciones Estratégicas</div>
            <div class="section-content">
                <ul class="recommendations">
                    {% for recommendation in report.recommendations %}
                    <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        
        <div style="text-align: center; padding: 20px; color: #666; font-size: 0.9em;">
            <p>Reporte generado automáticamente por el Sistema de Agentes IA Colaborativos</p>
            <p>{{ report.timestamp[:19] }}</p>
        </div>
    </div>
</body>
</html>
        """
        
        try:
            from jinja2 import Template
            template = Template(template_html)
            html_content = template.render(report=report)
            
            # Guardar archivo HTML
            filename = f"executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Reporte HTML generado: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error generando HTML: {e}")
    
    async def agent_logic(self):
        """Lógica principal del agente reporter"""
        current_time = time.time()
        
        if current_time - self.last_report_time >= self.report_interval:
            try:
                self.logger.info("Generando reporte ejecutivo...")
                report = await self.generate_executive_report()
                
                # Enviar notificación de reporte generado
                message = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    sender_type=self.agent_type,
                    message_type=MessageType.REPORT_REQUEST,
                    timestamp=datetime.now().isoformat(),
                    data={'report_id': report.id, 'summary': 'Reporte ejecutivo generado'}
                )
                
                await self.send_message(message)
                
                self.last_report_time = current_time
                self.logger.info(f"Reporte {report.id} generado exitosamente")
                
            except Exception as e:
                self.logger.error(f"Error generando reporte: {e}")

class CoordinatorAgent(BaseAgent):
    """Agente coordinador que orquesta la colaboración entre agentes"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.COORDINATOR, config)
        self.registered_agents = {}  # agent_id -> agent_info
        self.system_state = {}
        self.coordination_rules = self._load_coordination_rules()
    
    def _load_coordination_rules(self) -> Dict[str, Any]:
        """Cargar reglas de coordinación"""
        return {
            'alert_escalation': {
                'cpu_high': ['healer', 'reporter'],
                'memory_high': ['healer'],
                'disk_high': ['healer'],
                'service_down': ['healer', 'reporter']
            },
            'healing_timeout': 300,  # 5 minutos
            'max_concurrent_healings_per_system': 2,
            'report_triggers': ['critical_incident', 'healing_failure', 'scheduled']
        }
    
    async def register_agent(self, agent_info: Dict[str, Any]):
        """Registrar nuevo agente en el ecosistema"""
        agent_id = agent_info['agent_id']
        self.registered_agents[agent_id] = {
            **agent_info,
            'last_seen': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.logger.info(f"Agente registrado: {agent_id} ({agent_info['agent_type']})")
    
    async def handle_message(self, message: AgentMessage):
        """Coordinar mensajes entre agentes"""
        try:
            # Actualizar estado del agente
            if message.sender_id in self.registered_agents:
                self.registered_agents[message.sender_id]['last_seen'] = message.timestamp
            
            # Procesar según tipo de mensaje
            if message.message_type == MessageType.METRICS:
                await self._handle_metrics_message(message)
            
            elif message.message_type == MessageType.ALERT:
                await self._handle_alert_message(message)
            
            elif message.message_type == MessageType.HEALING_RESPONSE:
                await self._handle_healing_response(message)
            
        except Exception as e:
            self.logger.error(f"Error coordinando mensaje: {e}")
    
    async def _handle_metrics_message(self, message: AgentMessage):
        """Manejar métricas recibidas"""
        hostname = message.data.get('hostname', 'unknown')
        self.system_state[hostname] = {
            'last_metrics': message.data,
            'last_update': message.timestamp,
            'agent_id': message.sender_id
        }
    
    async def _handle_alert_message(self, message: AgentMessage):
        """Coordinar respuesta a alertas"""
        alert_data = message.data['alert']
        alert_type = alert_data['type']
        
        # Determinar qué agentes deben responder
        target_agents = self.coordination_rules['alert_escalation'].get(alert_type, ['healer'])
        
        # Filtrar agentes activos del tipo requerido
        active_targets = []
        for agent_id, agent_info in self.registered_agents.items():
            if any(agent_type in agent_info['agent_type'] for agent_type in target_agents):
                if agent_info['status'] == 'active':
                    active_targets.append(agent_id)
        
        if active_targets:
            # Reenviar alerta a agentes apropiados
            coordination_message = AgentMessage(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                sender_type=self.agent_type,
                message_type=MessageType.ALERT,
                timestamp=datetime.now().isoformat(),
                data=message.data,
                target_agents=active_targets,
                correlation_id=message.id
            )
            
            await self.send_message(coordination_message)
            self.logger.info(f"Alerta {alert_type} coordinada a {len(active_targets)} agentes")
        else:
            self.logger.warning(f"No hay agentes activos para manejar alerta {alert_type}")
    
    async def _handle_healing_response(self, message: AgentMessage):
        """Manejar respuestas de healing"""
        healing_data = message.data
        status = healing_data.get('status')
        
        if status == 'success':
            self.logger.info(f"Healing exitoso: {healing_data.get('healing_id')}")
        else:
            self.logger.warning(f"Healing falló: {healing_data.get('healing_id')}")
            
            # Trigger reporte si hay falla crítica
            await self._trigger_incident_report(healing_data)
    
    async def _trigger_incident_report(self, incident_data: Dict[str, Any]):
        """Activar generación de reporte por incidente"""
        reporter_agents = [
            agent_id for agent_id, agent_info in self.registered_agents.items()
            if 'reporter' in agent_info['agent_type'] and agent_info['status'] == 'active'
        ]
        
        if reporter_agents:
            report_message = AgentMessage(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                sender_type=self.agent_type,
                message_type=MessageType.REPORT_REQUEST,
                timestamp=datetime.now().isoformat(),
                data={
                    'trigger': 'critical_incident',
                    'incident_data': incident_data
                },
                target_agents=reporter_agents
            )
            
            await self.send_message(report_message)
    
    async def monitor_agent_health(self):
        """Monitorear salud de los agentes"""
        current_time = datetime.now()
        timeout_threshold = timedelta(minutes=5)
        
        for agent_id, agent_info in self.registered_agents.items():
            last_seen = datetime.fromisoformat(agent_info['last_seen'])
            
            if current_time - last_seen > timeout_threshold:
                if agent_info['status'] == 'active':
                    agent_info['status'] = 'timeout'
                    self.logger.warning(f"Agente {agent_id} no responde")
    
    async def agent_logic(self):
        """Lógica principal del coordinador"""
        # Monitorear salud de agentes cada minuto
        await self.monitor_agent_health()
        
        # Verificar estado general del ecosistema
        active_agents = sum(1 for agent in self.registered_agents.values() if agent['status'] == 'active')
        total_agents = len(self.registered_agents)
        
        if total_agents > 0 and active_agents / total_agents < 0.5:
            self.logger.warning(f"Solo {active_agents}/{total_agents} agentes activos en el ecosistema")

# Orquestador del ecosistema
class AgentEcosystem:
    """Orquestador principal del ecosistema de agentes"""
    
    def __init__(self, config_file: str = 'ecosystem_config.json'):
        self.config = self._load_config(config_file)
        self.agents = {}
        self.coordinator = None
        self.logger = logging.getLogger("ecosystem")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Cargar configuración del ecosistema"""
        default_config = {
            "agents": {
                "monitor_agents": 2,
                "healer_agents": 1,
                "reporter_agents": 1,
                "coordinator_agents": 1
            },
            "base_ports": {
                "monitor": 8100,
                "healer": 8200,
                "reporter": 8300,
                "coordinator": 8400
            },
            "monitoring": {
                "metrics_interval": 30,
                "report_interval": 3600,
                "thresholds": {
                    "cpu": 80,
                    "memory": 85,
                    "disk": 90
                }
            },
            "healing": {
                "max_concurrent_healings": 3,
                "timeout_seconds": 300
            },
            "critical_services": ["ssh", "nginx", "postgresql", "docker"]
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            self.logger.info(f"Creando configuración por defecto: {config_file}")
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    async def initialize_ecosystem(self):
        """Inicializar todo el ecosistema de agentes"""
        self.logger.info("🚀 Inicializando ecosistema de agentes colaborativos...")
        
        # Crear coordinador
        coordinator_config = {
            'ws_port': self.config['base_ports']['coordinator'],
            **self.config
        }
        self.coordinator = CoordinatorAgent('coordinator_001', coordinator_config)
        self.agents['coordinator_001'] = self.coordinator
        
        # Crear agentes monitores
        for i in range(self.config['agents']['monitor_agents']):
            agent_id = f'monitor_{i+1:03d}'
            monitor_config = {
                'ws_port': self.config['base_ports']['monitor'] + i,
                'metrics_interval': self.config['monitoring']['metrics_interval'],
                'thresholds': self.config['monitoring']['thresholds'],
                'critical_services': self.config['critical_services']
            }
            monitor = MonitorAgent(agent_id, monitor_config)
            self.agents[agent_id] = monitor
        
        # Crear agentes healers
        for i in range(self.config['agents']['healer_agents']):
            agent_id = f'healer_{i+1:03d}'
            healer_config = {
                'ws_port': self.config['base_ports']['healer'] + i,
                'max_concurrent_healings': self.config['healing']['max_concurrent_healings']
            }
            healer = HealerAgent(agent_id, healer_config)
            self.agents[agent_id] = healer
        
        # Crear agentes reporters
        for i in range(self.config['agents']['reporter_agents']):
            agent_id = f'reporter_{i+1:03d}'
            reporter_config = {
                'ws_port': self.config['base_ports']['reporter'] + i,
                'report_interval': self.config['monitoring']['report_interval']
            }
            reporter = ReporterAgent(agent_id, reporter_config)
            self.agents[agent_id] = reporter
        
        self.logger.info(f"✅ {len(self.agents)} agentes creados")
    
    async def connect_agents(self):
        """Conectar todos los agentes entre sí"""
        self.logger.info("🔗 Conectando agentes...")
        
        # Dar tiempo para que los servidores WebSocket se inicien
        await asyncio.sleep(2)
        
        agent_ports = {}
        for agent_id, agent in self.agents.items():
            agent_ports[agent_id] = agent.ws_port
        
        # Conectar cada agente con todos los demás
        for agent_id, agent in self.agents.items():
            for peer_id, peer_port in agent_ports.items():
                if peer_id != agent_id:
                    try:
                        await agent.connect_to_peer(peer_id, peer_port)
                    except Exception as e:
                        self.logger.error(f"Error conectando {agent_id} -> {peer_id}: {e}")
        
        self.logger.info("✅ Agentes conectados")
    
    async def start_ecosystem(self):
        """Iniciar todo el ecosistema"""
        try:
            # Inicializar agentes
            await self.initialize_ecosystem()
            
            # Iniciar todos los agentes
            tasks = []
            for agent_id, agent in self.agents.items():
                task = asyncio.create_task(agent.run())
                tasks.append(task)
                self.logger.info(f"🚀 Agente {agent_id} iniciado")
            
            # Esperar un poco y conectar agentes
            await asyncio.sleep(3)
            await self.connect_agents()
            
            # Registrar agentes en coordinador
            for agent_id, agent in self.agents.items():
                if agent_id != 'coordinator_001':
                    await self.coordinator.register_agent({
                        'agent_id': agent_id,
                        'agent_type': agent.agent_type.value,
                        'ws_port': agent.ws_port
                    })
            
            self.logger.info("🎉 ECOSISTEMA DE AGENTES COLABORATIVOS ACTIVO")
            self.logger.info("=" * 60)
            self.logger.info("🤖 Funcionalidades activas:")
            self.logger.info("   📊 Monitoreo continuo de sistemas")
            self.logger.info("   🔧 Auto-healing inteligente")
            self.logger.info("   📈 Reportes ejecutivos automáticos")
            self.logger.info("   🤝 Colaboración entre agentes")
            self.logger.info("   🧠 Coordinación inteligente")
            self.logger.info("=" * 60)
            
            # Esperar que terminen las tareas
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Deteniendo ecosistema...")
        except Exception as e:
            self.logger.error(f"Error en ecosistema: {e}")
            raise

# Ejemplo de uso y demostración
async def main():
    """Función principal para ejecutar el ecosistema"""
    
    print("🤖 ECOSISTEMA DE AGENTES IA COLABORATIVOS")
    print("=" * 50)
    print("🧠 Características avanzadas:")
    print("   🤝 Colaboración multi-agente")
    print("   🎛️  Auto-healing inteligente")
    print("   📊 Reportes ejecutivos automáticos")
    print("   🔄 Coordinación en tiempo real")
    print("   📈 Análisis predictivo")
    print("=" * 50)
    
    # Crear y configurar ecosistema
    ecosystem = AgentEcosystem()
    
    try:
        # Iniciar ecosistema completo
        await ecosystem.start_ecosystem()
        
    except KeyboardInterrupt:
        print("\n🛑 Ecosistema detenido por el usuario")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Configurar logging para demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ecosystem.log')
        ]
    )
    
    print("💻 INSTALACIÓN REQUERIDA:")
    print("pip install psutil websockets aiohttp plotly pandas jinja2 numpy scikit-learn")
    print("\n🚀 INICIANDO ECOSISTEMA...")
    print("Presiona Ctrl+C para detener")
    print("-" * 50)
    
    # Ejecutar ecosistema
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Ecosistema detenido limpiamente")
    except Exception as e:
        print(f"\n❌ Error ejecutando ecosistema: {e}")
        print("\n💡 Verifica que hayas instalado todas las dependencias:")
        print("pip install psutil websockets aiohttp plotly pandas jinja2 numpy scikit-learn") por sistema
        systems = {}
        
        for metric in self.collected_metrics:
            hostname = metric.get('hostname', 'unknown')
            if hostname not in systems:
                systems[hostname] = {
                    'cpu_values': [],
                    'memory_values': [],
                    'disk_values': [],
                    'last_seen': metric['timestamp']
                }
            
            systems[hostname]['cpu_values'].append(metric['cpu_percent'])
            systems[hostname]['memory_values'].append(metric['memory_percent'])
            systems[hostname]['disk_values'].append(metric['disk_percent'])
            systems[hostname]['last_seen'] = max(systems[hostname]['last_seen'], metric['timestamp'])
        
        # Calcular estadísticas