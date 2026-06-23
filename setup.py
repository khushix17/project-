#!/usr/bin/env python3
"""
🎯 Acoustic Eavesdropper - Setup Script
Automatically installs all dependencies and prepares the environment
"""

import os
import sys
import subprocess
import platform

class AcousticEavesdropperSetup:
    def __init__(self):
        self.colors = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'blue': '\033[94m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'reset': '\033[0m',
            'bold': '\033[1m'
        }
        self.os_name = platform.system()
        self.python_version = sys.version_info
        
    def print_header(self):
        """Print the setup header"""
        print(f"""
{self.colors['purple']}{self.colors['bold']}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     🎯 ACOUSTIC EAVESDROPPER - SETUP SCRIPT                     ║
║     Keyboard Sound Side-Channel Attack Tool                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
{self.colors['reset']}
        """)
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        print(f"{self.colors['blue']}📌 Checking Python version...{self.colors['reset']}")
        
        if self.python_version.major == 3 and self.python_version.minor >= 8:
            print(f"{self.colors['green']}✅ Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro} - OK{self.colors['reset']}")
            return True
        else:
            print(f"{self.colors['red']}❌ Python 3.8+ required. You have {self.python_version.major}.{self.python_version.minor}{self.colors['reset']}")
            return False
            
    def install_packages(self):
        """Install required Python packages"""
        print(f"\n{self.colors['blue']}📦 Installing required packages...{self.colors['reset']}")
        
        packages = [
            'torch',
            'torchaudio',
            'librosa',
            'numpy',
            'scipy',
            'matplotlib',
            'sounddevice',
            'pillow',
            'tqdm',
            'requests',
            'scikit-learn'
        ]
        
        success_count = 0
        total = len(packages)
        
        for i, package in enumerate(packages, 1):
            print(f"{self.colors['cyan']}[{i}/{total}] Installing {package}...{self.colors['reset']}")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
                print(f"{self.colors['green']}  ✅ {package} installed successfully{self.colors['reset']}")
                success_count += 1
            except subprocess.CalledProcessError:
                print(f"{self.colors['red']}  ❌ Failed to install {package}{self.colors['reset']}")
                
        print(f"\n{self.colors['green']}✅ Installed {success_count}/{total} packages successfully{self.colors['reset']}")
        return success_count == total
        
    def verify_installation(self):
        """Verify all packages are installed correctly"""
        print(f"\n{self.colors['blue']}🔍 Verifying installation...{self.colors['reset']}")
        
        packages_to_test = {
            'torch': 'torch',
            'librosa': 'librosa',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib',
            'sounddevice': 'sounddevice',
            'PIL': 'PIL'
        }
        
        all_ok = True
        for display_name, import_name in packages_to_test.items():
            try:
                __import__(import_name)
                print(f"{self.colors['green']}  ✅ {display_name}{self.colors['reset']}")
            except ImportError:
                print(f"{self.colors['red']}  ❌ {display_name} - NOT FOUND{self.colors['reset']}")
                all_ok = False
                
        return all_ok
        
    def run(self):
        """Main setup process"""
        self.print_header()
        
        # Check Python version
        if not self.check_python_version():
            print(f"{self.colors['red']}❌ Setup failed: Python version not compatible{self.colors['reset']}")
            sys.exit(1)
            
        # Install packages
        print(f"\n{self.colors['bold']}📦 Installing packages...{self.colors['reset']}")
        print(f"{self.colors['yellow']}⏳ This may take 2-5 minutes...{self.colors['reset']}")
        
        self.install_packages()
            
        # Verify installation
        if not self.verify_installation():
            print(f"{self.colors['red']}❌ Some packages are missing. Please check the errors above.{self.colors['reset']}")
            print(f"{self.colors['yellow']}💡 Try running: pip3 install torch librosa numpy matplotlib sounddevice pillow{self.colors['reset']}")
        else:
            print(f"\n{self.colors['purple']}{self.colors['bold']}✅ Setup Complete!{self.colors['reset']}")
            print(f"\n{self.colors['green']}🎉 All dependencies are installed!{self.colors['reset']}")
            print(f"\n{self.colors['bold']}🚀 TO GET STARTED:{self.colors['reset']}")
            print(f"   {self.colors['cyan']}python3 start.py{self.colors['reset']}")

if __name__ == "__main__":
    try:
        setup = AcousticEavesdropperSetup()
        setup.run()
    except KeyboardInterrupt:
        print(f"\n⚠️  Setup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)