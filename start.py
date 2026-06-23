cat > start.py << 'EOF'
#!/usr/bin/env python3
"""
🎯 Acoustic Eavesdropper - Keyboard Sound Attack Tool
A beautiful interface for acoustic side-channel attack demonstration
"""

import os
import sys
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import numpy as np
import sounddevice as sd
import wave
import glob

class AcousticEavesdropperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Acoustic Eavesdropper - Keyboard Attack Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0e27')
        
        # Variables
        self.is_recording = False
        self.training_status = "Not trained"
        self.current_model = None
        
        # Setup UI
        self.setup_styles()
        self.create_header()
        self.create_main_layout()
        self.create_status_bar()
        
        # Load initial status
        self.check_model_status()
        
    def setup_styles(self):
        """Setup modern dark theme colors"""
        self.colors = {
            'bg': '#0a0e27',
            'card': '#141833',
            'accent': '#6c5ce7',
            'accent2': '#00b894',
            'text': '#dfe6e9',
            'text_secondary': '#b2bec3',
            'danger': '#ff6b6b',
            'warning': '#fdcb6e',
            'success': '#00b894'
        }
        
    def create_header(self):
        """Create the header with title and status"""
        header_frame = tk.Frame(self.root, bg=self.colors['bg'])
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        # Left side - Title
        title_frame = tk.Frame(header_frame, bg=self.colors['bg'])
        title_frame.pack(side=tk.LEFT)
        
        title = tk.Label(title_frame, 
                         text="🎯 Acoustic Eavesdropper",
                         font=('Helvetica', 28, 'bold'),
                         fg=self.colors['accent'],
                         bg=self.colors['bg'])
        title.pack(side=tk.LEFT)
        
        subtitle = tk.Label(title_frame,
                           text="Keyboard Sound Side-Channel Attack",
                           font=('Helvetica', 12),
                           fg=self.colors['text_secondary'],
                           bg=self.colors['bg'])
        subtitle.pack(side=tk.LEFT, padx=(15, 0))
        
        # Right side - Status indicator
        status_frame = tk.Frame(header_frame, bg=self.colors['bg'])
        status_frame.pack(side=tk.RIGHT)
        
        self.status_indicator = tk.Label(status_frame,
                                         text="🔴 Not Ready",
                                         font=('Helvetica', 12, 'bold'),
                                         fg=self.colors['danger'],
                                         bg=self.colors['bg'])
        self.status_indicator.pack(side=tk.LEFT, padx=10)
        
        self.model_status = tk.Label(status_frame,
                                     text="Model: ❌ Not trained",
                                     font=('Helvetica', 10),
                                     fg=self.colors['text_secondary'],
                                     bg=self.colors['bg'])
        self.model_status.pack(side=tk.LEFT, padx=10)
        
        # Separator
        separator = tk.Frame(self.root, height=2, bg=self.colors['accent'])
        separator.pack(fill=tk.X, padx=20, pady=5)
        
    def create_main_layout(self):
        """Create the main content area"""
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Panel - Controls
        left_panel = tk.Frame(main_container, bg=self.colors['card'], relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left_panel.configure(width=400)
        left_panel.pack_propagate(False)
        
        # Right Panel - Output/Visualization
        right_panel = tk.Frame(main_container, bg=self.colors['card'], relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ----- LEFT PANEL CONTENT -----
        self.create_control_panel(left_panel)
        
        # ----- RIGHT PANEL CONTENT -----
        self.create_output_panel(right_panel)
        
    def create_control_panel(self, parent):
        """Create the control panel with all buttons"""
        # Title
        title = tk.Label(parent,
                        text="🎮 Control Center",
                        font=('Helvetica', 16, 'bold'),
                        fg=self.colors['text'],
                        bg=self.colors['card'])
        title.pack(pady=(15, 10), padx=10)
        
        # Separator
        sep = tk.Frame(parent, height=2, bg=self.colors['accent'])
        sep.pack(fill=tk.X, padx=10, pady=5)
        
        # Step 1: Record Training Data
        self.create_section(parent, "📝 Step 1: Record Training Data",
                           "Record keystrokes for training the model",
                           self.record_training,
                           "🎙️ Record Training Data")
        
        # Step 2: Train Model
        self.create_section(parent, "🧠 Step 2: Train Model",
                           "Train neural network on recorded keystrokes",
                           self.train_model,
                           "⚡ Train Model")
        
        # Step 3: Record Test
        self.create_section(parent, "🎤 Step 3: Record Test",
                           "Record a test phrase to attack",
                           self.record_test,
                           "📼 Record Test")
        
        # Step 4: Run Attack
        self.create_section(parent, "💀 Step 4: Run Attack",
                           "Decode the recorded test phrase",
                           self.run_attack,
                           "🔓 Run Attack")
        
        # Progress section
        progress_frame = tk.Frame(parent, bg=self.colors['card'])
        progress_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(progress_frame,
                text="📊 Progress",
                font=('Helvetica', 12, 'bold'),
                fg=self.colors['text'],
                bg=self.colors['card']).pack(anchor=tk.W)
        
        self.progress = ttk.Progressbar(progress_frame, 
                                       length=350, 
                                       mode='determinate',
                                       style="green.Horizontal.TProgressbar")
        self.progress.pack(pady=5)
        
        self.progress_label = tk.Label(progress_frame,
                                      text="Ready",
                                      font=('Helvetica', 9),
                                      fg=self.colors['text_secondary'],
                                      bg=self.colors['card'])
        self.progress_label.pack()
        
    def create_section(self, parent, title, desc, command, button_text):
        """Create a section with title, description, and button"""
        frame = tk.Frame(parent, bg=self.colors['card'])
        frame.pack(fill=tk.X, padx=15, pady=8)
        
        # Title
        tk.Label(frame,
                text=title,
                font=('Helvetica', 11, 'bold'),
                fg=self.colors['text'],
                bg=self.colors['card']).pack(anchor=tk.W)
        
        # Description
        tk.Label(frame,
                text=desc,
                font=('Helvetica', 9),
                fg=self.colors['text_secondary'],
                bg=self.colors['card']).pack(anchor=tk.W)
        
        # Button
        btn = tk.Button(frame,
                       text=button_text,
                       command=command,
                       font=('Helvetica', 10, 'bold'),
                       bg=self.colors['accent'],
                       fg='white',
                       relief=tk.FLAT,
                       cursor='hand2',
                       padx=20,
                       pady=8)
        btn.pack(pady=(5, 0))
        
        # Hover effect
        btn.bind('<Enter>', lambda e, b=btn: b.config(bg='#7c6ce7'))
        btn.bind('<Leave>', lambda e, b=btn: b.config(bg=self.colors['accent']))
        
    def create_output_panel(self, parent):
        """Create the output panel with visualization"""
        # Notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Console Output
        console_tab = tk.Frame(self.notebook, bg=self.colors['card'])
        self.notebook.add(console_tab, text="📋 Console")
        
        self.console = scrolledtext.ScrolledText(console_tab,
                                                 wrap=tk.WORD,
                                                 font=('Courier', 10),
                                                 bg='#1a1a2e',
                                                 fg='#00ff41',
                                                 insertbackground='#00ff41',
                                                 height=20)
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.console.insert('1.0', "🎯 Acoustic Eavesdropper Started...\n")
        self.console.insert('end', "📌 Waiting for user action...\n\n")
        
        # Tab 2: Visualization
        viz_tab = tk.Frame(self.notebook, bg=self.colors['card'])
        self.notebook.add(viz_tab, text="📊 Visualization")
        
        # Matplotlib figure
        self.fig = Figure(figsize=(6, 4), dpi=80, facecolor='#141833')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a1a2e')
        self.ax.tick_params(colors='#dfe6e9')
        for spine in self.ax.spines.values():
            spine.set_color('#dfe6e9')
        self.ax.set_xlabel('Time', color='#dfe6e9')
        self.ax.set_ylabel('Amplitude', color='#dfe6e9')
        self.ax.set_title('Audio Waveform', color='#dfe6e9')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 3: Results
        results_tab = tk.Frame(self.notebook, bg=self.colors['card'])
        self.notebook.add(results_tab, text="🎯 Results")
        
        self.results_text = scrolledtext.ScrolledText(results_tab,
                                                     wrap=tk.WORD,
                                                     font=('Helvetica', 14, 'bold'),
                                                     bg='#1a1a2e',
                                                     fg='#00ff41',
                                                     height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.results_text.insert('1.0', "🔍 Results will appear here...\n")
        self.results_text.config(state='disabled')
        
    def create_status_bar(self):
        """Create the status bar at the bottom"""
        status_bar = tk.Frame(self.root, bg=self.colors['card'], height=30)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(0, 10))
        
        self.status_text = tk.Label(status_bar,
                                    text="✅ System Ready",
                                    font=('Helvetica', 10),
                                    fg=self.colors['text_secondary'],
                                    bg=self.colors['card'])
        self.status_text.pack(side=tk.LEFT, padx=10)
        
        self.timer_label = tk.Label(status_bar,
                                   text="⏱️ Ready",
                                   font=('Helvetica', 10),
                                   fg=self.colors['text_secondary'],
                                   bg=self.colors['card'])
        self.timer_label.pack(side=tk.RIGHT, padx=10)
        
    def log(self, message, status='info'):
        """Add message to console"""
        self.console.insert('end', f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.console.see('end')
        self.root.update()
        
        # Update status bar
        if status == 'error':
            self.status_text.config(fg=self.colors['danger'], text=f"❌ {message}")
        elif status == 'success':
            self.status_text.config(fg=self.colors['success'], text=f"✅ {message}")
        else:
            self.status_text.config(fg=self.colors['text_secondary'], text=f"ℹ️ {message}")
            
    def update_progress(self, value, label):
        """Update progress bar"""
        self.progress['value'] = value
        self.progress_label.config(text=label)
        self.root.update()
        
    def check_model_status(self):
        """Check if a trained model exists"""
        if os.path.exists('modelo_teclado.pth') or os.path.exists('modelo_simple.pkl'):
            self.model_status.config(text="Model: ✅ Trained")
            self.status_indicator.config(text="🟢 Ready", fg=self.colors['success'])
            self.log("✅ Trained model found!", 'success')
        else:
            self.model_status.config(text="Model: ❌ Not trained")
            self.status_indicator.config(text="🟡 Needs Training", fg=self.colors['warning'])
            self.log("⚠️ No trained model found. Please record training data first.", 'warning')
            
    def record_training(self):
        """Record training data"""
        self.log("🎙️ Starting training data recording...", 'info')
        self.update_progress(10, "Recording training data...")
        
        def run():
            try:
                # Check if quickstart.py exists
                if not os.path.exists('quickstart.py'):
                    self.log("❌ quickstart.py not found!", 'error')
                    return
                
                # Run the recording script
                self.log("📝 Follow the prompts in the terminal...")
                process = subprocess.Popen(['python3', 'quickstart.py'],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         text=True)
                
                # Read output in real-time
                for line in process.stdout:
                    self.log(f"  {line.strip()}")
                
                process.wait()
                
                if process.returncode == 0:
                    self.log("✅ Training data recorded successfully!", 'success')
                    self.update_progress(30, "Recording complete!")
                    self.check_model_status()
                else:
                    self.log("❌ Recording failed!", 'error')
                    self.update_progress(0, "Failed")
                    
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", 'error')
                
        threading.Thread(target=run, daemon=True).start()
        
    def train_model(self):
        """Train the model"""
        self.log("🧠 Starting model training...", 'info')
        self.update_progress(40, "Training model...")
        
        def run():
            try:
                # Check if training data exists
                if not os.path.exists('dados_treino'):
                    self.log("❌ No training data found! Please record training data first.", 'error')
                    self.update_progress(0, "No data found")
                    return
                
                # Try to use the working training script
                if os.path.exists('train_working.py'):
                    script = 'train_working.py'
                else:
                    script = 'hacker_de_teclado.py'
                
                self.log(f"📚 Using {script} for training...")
                process = subprocess.Popen(['python3', script, '--treinar', 'dados_treino/'],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         text=True)
                
                for line in process.stdout:
                    if 'Epoch' in line or 'Training' in line:
                        self.log(f"  {line.strip()}")
                    if 'complete' in line.lower():
                        self.log("✅ Model training complete!", 'success')
                        self.update_progress(80, "Training complete!")
                        self.check_model_status()
                
                process.wait()
                
                if process.returncode == 0:
                    self.log("✅ Training completed successfully!", 'success')
                    self.update_progress(100, "Done!")
                    self.show_results("🎯 Model Training Complete!\n\n✅ Model saved as modelo_teclado.pth\n✅ Ready for testing!")
                else:
                    self.log("❌ Training failed!", 'error')
                    self.update_progress(0, "Failed")
                    
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", 'error')
                
        threading.Thread(target=run, daemon=True).start()
        
    def record_test(self):
        """Record a test phrase"""
        self.log("🎤 Recording test phrase...", 'info')
        self.update_progress(60, "Recording test...")
        
        def run():
            try:
                if not os.path.exists('record_test.py'):
                    self.log("❌ record_test.py not found!", 'error')
                    return
                
                self.log("📝 Type 'o zorro e gris' during the 8-second recording...")
                self.log("⏳ Recording will start in 3 seconds...")
                
                process = subprocess.Popen(['python3', 'record_test.py'],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         text=True)
                
                for line in process.stdout:
                    if 'Recording' in line or 'Saved' in line:
                        self.log(f"  {line.strip()}")
                
                process.wait()
                
                if process.returncode == 0:
                    self.log("✅ Test recording saved!", 'success')
                    self.update_progress(80, "Test recorded!")
                else:
                    self.log("❌ Recording failed!", 'error')
                    self.update_progress(60, "Failed")
                    
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", 'error')
                
        threading.Thread(target=run, daemon=True).start()
        
    def run_attack(self):
        """Run the attack on the test recording"""
        self.log("💀 Running attack...", 'info')
        self.update_progress(80, "Decoding keystrokes...")
        
        def run():
            try:
                test_file = 'meu_teste_zorro.wav'
                if not os.path.exists(test_file):
                    self.log("❌ No test recording found! Please record a test first.", 'error')
                    self.update_progress(0, "No test file")
                    return
                
                # Try to use the main script
                if os.path.exists('hacker_de_teclado.py'):
                    process = subprocess.Popen(['python3', 'hacker_de_teclado.py', '--prever', test_file],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,
                                             text=True)
                    
                    output_lines = []
                    for line in process.stdout:
                        self.log(f"  {line.strip()}")
                        if 'Predicted' in line:
                            output_lines.append(line.strip())
                    
                    process.wait()
                    
                    if output_lines:
                        # Show results
                        result_text = "🎯 ATTACK RESULTS\n" + "="*50 + "\n\n"
                        for line in output_lines:
                            result_text += f"🔓 {line}\n"
                        self.show_results(result_text)
                        self.log("✅ Attack completed!", 'success')
                        self.update_progress(100, "Done!")
                    else:
                        self.log("⚠️ No predictions made. Check the test recording.", 'warning')
                        
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", 'error')
                
        threading.Thread(target=run, daemon=True).start()
        
    def show_results(self, text):
        """Display results in the results tab"""
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', text)
        self.results_text.config(state='disabled')
        self.notebook.select(2)  # Switch to results tab
        
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()

# Main entry point
if __name__ == "__main__":
    # Check for required packages
    try:
        import PIL
        import matplotlib
        import librosa
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📦 Run: pip3 install pillow matplotlib librosa sounddevice")
        sys.exit(1)
    
    root = tk.Tk()
    app = AcousticEavesdropperGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
EOF