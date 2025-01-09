#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
from pathlib import Path
import asyncio
import ipaddress
from typing import Optional, Tuple, Dict, List, Any
import socket
import netifaces
import yaml
from datetime import datetime, timedelta
import dns.resolver
import json
from dataclasses import dataclass
import aiodns
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from aiohttp import ClientSession
import logging
import bcrypt
from functools import lru_cache
import multiprocessing
import psutil
import time
import uuid
from cachetools import TTLCache
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey, 
    BigInteger, Index, text, JSONB, UUID, create_engine, select
)
from sqlalchemy.orm import (
    declarative_base, relationship, sessionmaker
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from redis.asyncio import Redis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("IKEv2 Installer")

def check_root():
    """Check if the script is running as root."""
    if os.geteuid() != 0:
        logger.error("This script must be run as root! Please run with sudo or as root user.")
        sys.exit(1)

# Version check
MIN_PYTHON = (3, 9)
if sys.version_info < MIN_PYTHON:
    sys.exit(f"Python {'.'.join(str(n) for n in MIN_PYTHON)} or later required.")

check_root()

class DependencyManager:
    def __init__(self, venv_path="/opt/my_module_venv"):
        self.venv_path = Path(venv_path)
        self.venv_python = self.venv_path / "bin" / "python"
        self.venv_pip = self.venv_path / "bin" / "pip"

        # Complete package list with version constraints
        self.required_packages = [
            # Core dependencies
            "redis>=5.2.0,<6.0.0",
            "asyncpg>=0.30.0,<0.31.0",
            "sqlalchemy>=2.0.0,<3.0.0",
            "fastapi>=0.95.0,<1.0.0",
            "uvicorn>=0.34.0,<0.35.0",
            
            # Security and authentication
            "cryptography>=41.0.0,<42.0.0",
            "bcrypt>=4.0.0,<5.0.0",
            "passlib>=1.7.4,<2.0.0",
            "python-jose[cryptography]>=3.3.0,<4.0.0",
            "python-multipart>=0.0.6,<0.1.0",
            
            # Data validation and API
            "pydantic>=2.0.0,<3.0.0",
            "email-validator>=2.0.0,<3.0.0",
            
            # System monitoring and metrics
            "psutil>=5.9.0,<6.0.0",
            "prometheus_client>=0.17.0,<0.18.0",
            
            # Network and VPN specific
            "pyroute2>=0.7.0,<0.8.0",
            "netaddr>=0.8.0,<0.9.0",
            "pyOpenSSL>=23.0.0,<24.0.0",
            "aiodns>=3.0.0,<4.0.0",
            "netifaces>=0.11.0,<0.12.0",
            "pyyaml>=6.0.1,<7.0.0",
            "aiohttp>=3.8.0,<4.0.0",
            "dnspython>=2.3.0,<3.0.0",
            
            # Automation and deployment
            "ansible>=8.0.0,<9.0.0",
            "docker>=6.0.0,<7.0.0",
            
            # Testing and development
            "pytest>=7.0.0,<8.0.0",
            "pytest-asyncio>=0.21.0,<0.22.0",
            "pytest-cov>=4.0.0,<5.0.0"
        ]

        # System dependencies
        self.system_packages = [
            # VPN and security
            "strongswan",
            "strongswan-pki",
            "libstrongswan-extra-plugins",
            "iptables",
            "netfilter-persistent",
            "wireguard",
            
            # Build essentials and development
            "build-essential",
            "python3-dev",
            "libpq-dev",
            "libffi-dev",
            "libssl-dev",
            
            # Database
            "postgresql",
            "postgresql-contrib",
            
            # Network tools
            "net-tools",
            "iproute2",
            "tcpdump",
            "iputils-ping",
            "dnsutils",
            
            # Performance and monitoring
            "sysstat",
            "linux-tools-generic",
            
            # Hardware acceleration
            "intel-microcode",
            "aesni-intel",
            "linux-modules-extra-$(uname -r)"
        ]

    def check_system_dependencies(self):
        """Check and verify system package dependencies."""
        try:
            # Update package list
            subprocess.check_call(["apt-get", "update"])
            
            # Check kernel modules
            required_modules = ['af_key', 'ah4', 'ah6', 'esp4', 'esp6', 'xfrm_user']
            loaded_modules = subprocess.check_output(['lsmod']).decode()
            
            for module in required_modules:
                if module not in loaded_modules:
                    try:
                        subprocess.check_call(['modprobe', module])
                        logger.info(f"Loaded kernel module: {module}")
                    except subprocess.CalledProcessError:
                        logger.error(f"Failed to load kernel module: {module}")
                        return False

            # Install system packages
            for package in self.system_packages:
                try:
                    result = subprocess.run(
                        ["dpkg", "-l", package],
                        capture_output=True,
                        text=True
                    )
                    if package not in result.stdout:
                        logger.info(f"Installing {package}...")
                        subprocess.check_call(["apt-get", "install", "-y", package])
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {package}: {e}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error checking system dependencies: {e}")
            return False

    def setup_virtualenv(self):
        """Set up a Python virtual environment and install dependencies."""
        try:
            if not self.venv_path.exists():
                logger.info(f"Creating virtual environment at {self.venv_path}...")
                subprocess.check_call([sys.executable, "-m", "venv", str(self.venv_path)])

            # Upgrade pip and setuptools
            logger.info("Upgrading pip and setuptools...")
            subprocess.check_call([str(self.venv_pip), "install", "--upgrade", "pip", "setuptools", "wheel"])

            # Install required packages
            total_packages = len(self.required_packages)
            for i, package in enumerate(self.required_packages, 1):
                try:
                    logger.info(f"Installing package [{i}/{total_packages}]: {package}")
                    subprocess.check_call([str(self.venv_pip), "install", package])
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {package}: {e}")
                    sys.exit(1)

            logger.info(f"Virtual environment setup complete at {self.venv_python}")
        except Exception as e:
            logger.error(f"Error during virtual environment setup: {e}")
            sys.exit(1)

# Database Models
Base = declarative_base()

class User(Base):
    """User model with PostgreSQL-specific optimizations."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    user_metadata = Column(JSONB, nullable=True)

    __table_args__ = (
        Index('idx_user_username_email', 'username', 'email'),
        Index('idx_user_created_at', text('created_at DESC')),
        Index('idx_user_metadata', 'user_metadata', postgresql_using='gin')
    )

    vpn_certificates = relationship("VPNCertificate", back_populates="user")
    vpn_sessions = relationship("VPNSession", back_populates="user")

class VPNCertificate(Base):
    """VPN certificate model with optimized storage."""
    __tablename__ = "vpn_certificates"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    certificate_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    certificate_data = Column(Text, nullable=False)
    private_key = Column(Text, nullable=False)
    issued_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSONB, nullable=True)

    __table_args__ = (
        Index('idx_cert_user_id', 'user_id'),
        Index('idx_cert_expiry', 'expires_at'),
        Index('idx_cert_status', 'is_active', 'revoked_at'),
        Index('idx_cert_metadata', 'metadata', postgresql_using='gin')
    )

    user = relationship("User", back_populates="vpn_certificates")

class VPNSession(Base):
    """VPN session model with performance optimizations."""
    __tablename__ = "vpn_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    ip_address = Column(String, nullable=False)
    connected_at = Column(DateTime, default=datetime.utcnow)
    disconnected_at = Column(DateTime, nullable=True)
    bytes_sent = Column(BigInteger, default=0)
    bytes_received = Column(BigInteger, default=0)
    status = Column(String, nullable=False)
    connection_info = Column(JSONB, nullable=True)
    protocol = Column(String, nullable=True)
    encryption_type = Column(String, nullable=True)
    mtu = Column(Integer, nullable=True)
    qos_profile = Column(String, nullable=True)
    region = Column(String, nullable=True)

    __table_args__ = (
        Index('idx_session_user_id', 'user_id'),
        Index('idx_session_status', 'status'),
        Index('idx_session_timing', 'connected_at', 'disconnected_at'),
        Index('idx_session_conn_info', 'connection_info', postgresql_using='gin')
    )

    user = relationship("User", back_populates="vpn_sessions")

class DatabaseManager:
    """Database manager with PostgreSQL optimizations."""
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.engine = None
        self.sessionmaker = None
        
        # Database metrics
        self.metrics = {
            'db_connections': Gauge(
                'vpn_db_connections',
                'Active database connections',
                ['pool_type']
            ),
            'query_duration': Histogram(
                'vpn_db_query_duration_seconds',
                'Database query duration',
                ['query_type']
            ),
            'query_errors': Counter(
                'vpn_db_query_errors_total',
                'Database query errors',
                ['error_type']
            )
        }

    async def initialize(self):
        """Initialize database with optimized settings."""
        try:
            # Create async engine with optimized settings
            self.engine = create_async_engine(
                self.config['DATABASE_URL'],
                echo=self.config.get('DEBUG', False),
                pool_size=20,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True
            )
            
            # Create session factory
            self.sessionmaker = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.logger.info("Database initialization successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False

    @lru_cache(maxsize=1000)
    async def get_cached_query(self, query_key: str):
        """Get cached query result."""
        return await self._execute_cached_query(query_key)

    async def get_session(self) -> AsyncSession:
        """Get database session with monitoring."""
        try:
            session = self.sessionmaker()
            self.metrics['db_connections'].labels(pool_type='session').inc()
            return session
        except Exception as e:
            self.logger.error(f"Failed to create database session: {e}")
            self.metrics['query_errors'].labels(error_type='session_creation').inc()
            raise

@dataclass
class ServerConfig:
    """Server configuration."""
    domain: Optional[str]
    ipv6_address: Optional[str]
    ipv4_address: Optional[str]
    cert_path: str
    key_path: str
    lifetime_days: int

class NetworkManager:
    """Network management system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'network_throughput': Gauge(
                'vpn_network_throughput_bytes',
                'Network throughput in bytes/sec',
                ['interface', 'direction']
            ),
            'latency': Histogram(
                'vpn_network_latency_seconds',
                'Network latency in seconds',
                ['destination']
            ),
            'packet_loss': Counter(
                'vpn_packet_loss_total',
                'Total packet loss count',
                ['interface']
            ),
            'connection_count': Gauge(
                'vpn_active_connections',
                'Number of active network connections'
            )
        }
        self.interface_manager = InterfaceManager()
        self.route_manager = RouteManager()
        self.firewall_manager = FirewallManager()
        
    async def initialize(self):
        """Initialize network management system."""
        try:
            # Initialize components in parallel
            await asyncio.gather(
                self.interface_manager.initialize(),
                self.route_manager.initialize(),
                self.firewall_manager.initialize()
            )
            
            # Configure network optimizations
            await self._configure_optimizations()
            
            # Start monitoring
            await self._start_monitoring()
            
            self.logger.info("Network management system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Network initialization failed: {e}")
            raise

    async def _configure_optimizations(self):
        """Configure network optimizations."""
        try:
            # System network parameters
            sysctl_params = {
                'net.ipv4.ip_forward': 1,
                'net.ipv4.conf.all.accept_redirects': 0,
                'net.ipv4.conf.all.send_redirects': 0,
                'net.ipv4.tcp_congestion_control': 'bbr',
                'net.core.rmem_max': 16777216,
                'net.core.wmem_max': 16777216,
                'net.ipv4.tcp_rmem': '4096 87380 16777216',
                'net.ipv4.tcp_wmem': '4096 87380 16777216'
            }
            
            for param, value in sysctl_params.items():
                await self._set_sysctl(param, value)
                
        except Exception as e:
            self.logger.error(f"Network optimization failed: {e}")
            raise

    async def _start_monitoring(self):
        """Start network monitoring."""
        try:
            asyncio.create_task(self._monitor_network_metrics())
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            raise

    async def _monitor_network_metrics(self):
        """Monitor network metrics continuously."""
        while True:
            try:
                # Collect network metrics
                throughput = await self._collect_throughput()
                latency = await self._collect_latency()
                packet_loss = await self._collect_packet_loss()
                connections = await self._collect_connections()
                
                # Update Prometheus metrics
                self._update_metrics({
                    'throughput': throughput,
                    'latency': latency,
                    'packet_loss': packet_loss,
                    'connections': connections
                })
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Metric collection failed: {e}")
                await asyncio.sleep(5)  # Wait before retry

class InterfaceManager:
    """Manage network interfaces."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize interface management."""
        try:
            # Configure interfaces
            await self._configure_interfaces()
            return True
        except Exception as e:
            self.logger.error(f"Interface initialization failed: {e}")
            raise

class RouteManager:
    """Manage network routing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize routing management."""
        try:
            # Configure routing
            await self._configure_routing()
            return True
        except Exception as e:
            self.logger.error(f"Routing initialization failed: {e}")
            raise

class FirewallManager:
    """Manage firewall rules."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize firewall management."""
        try:
            # Configure firewall
            await self._configure_firewall()
            return True
        except Exception as e:
            self.logger.error(f"Firewall initialization failed: {e}")
            raise

class VPNManager:
    """Manage VPN connections and configurations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'vpn_connections': Gauge(
                'vpn_active_connections_total',
                'Total active VPN connections'
            ),
            'connection_errors': Counter(
                'vpn_connection_errors_total',
                'Total VPN connection errors',
                ['error_type']
            ),
            'bandwidth_usage': Gauge(
                'vpn_bandwidth_usage_bytes',
                'VPN bandwidth usage in bytes',
                ['direction']
            )
        }
        
    async def initialize(self):
        """Initialize VPN management system."""
        try:
            # Configure VPN
            await self._configure_vpn()
            
            # Start monitoring
            await self._start_vpn_monitoring()
            
            self.logger.info("VPN management system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"VPN initialization failed: {e}")
            raise

    async def _configure_vpn(self):
        """Configure VPN settings."""
        try:
            # Load VPN configuration
            config = await self._load_vpn_config()
            
            # Apply configuration
            await self._apply_vpn_config(config)
            
            # Setup monitoring
            await self._setup_vpn_monitoring()
            
        except Exception as e:
            self.logger.error(f"VPN configuration failed: {e}")
            raise

    async def _start_vpn_monitoring(self):
        """Start VPN monitoring."""
        try:
            asyncio.create_task(self._monitor_vpn_metrics())
        except Exception as e:
            self.logger.error(f"Failed to start VPN monitoring: {e}")
            raise

    async def _monitor_vpn_metrics(self):
        """Monitor VPN metrics continuously."""
        while True:
            try:
                # Collect VPN metrics
                connections = await self._collect_vpn_connections()
                bandwidth = await self._collect_bandwidth_usage()
                
                # Update Prometheus metrics
                self.metrics['vpn_connections'].set(connections)
                self.metrics['bandwidth_usage'].labels(direction='in').set(bandwidth['in'])
                self.metrics['bandwidth_usage'].labels(direction='out').set(bandwidth['out'])
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                self.logger.error(f"VPN metric collection failed: {e}")
                await asyncio.sleep(5)  # Wait before retry

if __name__ == "__main__":
    async def main():
        try:
            # Initialize managers
            network_manager = NetworkManager()
            vpn_manager = VPNManager()
            
            # Initialize systems in parallel
            await asyncio.gather(
                network_manager.initialize(),
                vpn_manager.initialize()
            )
            
            # Start Prometheus metrics server
            start_http_server(8000)
            
            # Keep the script running
            while True:
                await asyncio.sleep(3600)
                
        except Exception as e:
            logging.error(f"System initialization failed: {e}")
            sys.exit(1)
    
    # Run main function
    asyncio.run(main())
