#!/usr/bin/env python3
"""
Monitoring system for Who Dey Tallk
Provides visual feedback on system performance and conversation detection
"""

import time
import threading
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
import platform
import queue

logger = logging.getLogger('who_dey_tallk.monitor')

class SystemMonitor:
    """
    Visual monitoring system for Who Dey Tallk
    Displays system status, active components, and detected conversations
    """

    def __init__(self, refresh_rate=1.0):
        """
        Initialize the system monitor
        
        Args:
            refresh_rate: How often to refresh the display (in seconds)
        """
        self.refresh_rate = refresh_rate
        self.running = False
        self.threads = []
        self.components = {}
        self.conversation_history = []
        self.update_queue = queue.Queue()
        
        # Create the root window immediately
        # On macOS, this should only be called from the main thread
        self.root = tk.Tk()
        self.root.title("Who Dey Tallk - System Monitor")
        self.root.geometry("800x600")
        
        # Initialize status indicators dictionary
        self.status_indicators = {}
        self.conversation_display = None
        
        # Setup the UI elements
        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI elements for the monitor"""
        # Create frame for system status
        status_frame = ttk.LabelFrame(self.root, text="System Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create frame for conversation display
        conversation_frame = ttk.LabelFrame(self.root, text="Conversation History")
        conversation_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Add conversation text display
        self.conversation_display = scrolledtext.ScrolledText(
            conversation_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=20
        )
        self.conversation_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add tags for formatting
        self.conversation_display.tag_configure("speaker", font=("Arial", 10, "bold"))
        self.conversation_display.tag_configure("transcript", font=("Arial", 10))
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)
        # Store app controller reference
        self.app_controller = None

    def set_components(self, components):
        """Store component references and setup status indicators."""
        logger.debug("Setting components for monitor")
        self.components = components
        # Store app controller if passed
        if "app_controller" in components:
            self.app_controller = components["app_controller"]
            # Don't display the controller itself in the status
            # del components["app_controller"] # Or just skip it in _setup_status_indicators

        # Setup status indicators now that components are known
        # Ensure this runs on the main thread if called after root is created
        if self.root and self.root.winfo_exists():
            self.root.after(0, self._setup_status_indicators)
        else:
            # Should ideally not happen if called after __init__
            self._setup_status_indicators()

    def start(self):
        """
        Start the monitoring system's background tasks.
        Assumes set_components has already been called.
        """
        if self.running:
            logger.warning("Monitoring system already running")
            return

        if not self.components:
             logger.warning("Components not set before starting monitor. Status indicators may be empty.")
             # Still proceed, maybe components will be set later?

        self.running = True

        # Start refresh thread
        refresh_thread = threading.Thread(target=self._refresh_thread, daemon=True, name="MonitorRefreshThread")
        refresh_thread.start()
        self.threads.append(refresh_thread)

        # Initialize conversation display
        self._update_conversation_display()

        # Start checking for UI updates from the main thread
        self._check_update_queue()

        logger.info("Monitoring background tasks started")

    def _check_update_queue(self):
        """Process any pending UI updates from the queue"""
        if not self.running or not self.root or not self.root.winfo_exists():
            return
            
        try:
            # Process all available updates
            while True:
                update_func, args = self.update_queue.get_nowait()
                update_func(*args)
                self.update_queue.task_done()
        except queue.Empty:
            pass
        
        # Schedule next check if still running
        if self.running and self.root and self.root.winfo_exists():
            self.root.after(100, self._check_update_queue)

    def _setup_status_indicators(self):
        """Set up status indicators for each component"""
        # Find status frame
        status_frame = None
        for child in self.root.winfo_children():
            if isinstance(child, ttk.LabelFrame) and child.cget("text") == "System Status":
                status_frame = child
                break
        
        if not status_frame:
            logger.error("Status frame not found")
            return
            
        # Clear existing status indicators
        for widget in status_frame.winfo_children():
            widget.destroy()
        
        # Add status indicators for each component
        self.status_indicators = {}
        row = 0
        for comp_name in self.components.keys():
            label = ttk.Label(status_frame, text=f"{comp_name.replace('_', ' ').title()}:")
            label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            
            status = ttk.Label(status_frame, text="Initializing...", foreground="orange")
            status.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            
            self.status_indicators[comp_name] = status
            row += 1

    def stop(self):
        """Stop the monitoring system and destroy the UI window."""
        if not self.running:
            return

        logger.debug("Stopping monitoring system...")
        self.running = False

        # Wait for background refresh thread to finish
        logger.debug(f"Waiting for {len(self.threads)} monitor threads to join...")
        for thread in self.threads:
            if thread.is_alive():
                logger.debug(f"Joining thread: {thread.name}")
                thread.join(timeout=1.0)
                if thread.is_alive():
                    logger.warning(f"Monitor thread {thread.name} did not exit cleanly.")
        self.threads = []

        # Destroy the UI window if it exists
        # This needs to happen on the main thread
        def destroy_ui():
            if self.root and self.root.winfo_exists():
                logger.debug("Destroying monitor UI window.")
                self.root.destroy()
            self.root = None # Ensure reference is cleared

        if self.root:
            try:
                # Schedule the destroy call on the main thread
                self.root.after(0, destroy_ui)
            except tk.TclError:
                 # Window might already be destroyed
                 self.root = None

        logger.info("Monitoring system stopped")

    def update_conversation(self, speaker_id, speaker_name, confidence, transcript):
        """
        Update the conversation display with new speech
        
        Args:
            speaker_id: ID of the speaker
            speaker_name: Name of the speaker (or "Unknown")
            confidence: Confidence score for the speaker identification
            transcript: Transcribed speech text
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = {
            'timestamp': timestamp,
            'speaker_id': speaker_id,
            'speaker_name': speaker_name,
            'confidence': confidence,
            'transcript': transcript
        }
        
        self.conversation_history.append(entry)
        
        # Limit history to latest 100 entries
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
            
        # Queue UI update instead of direct update
        self.update_queue.put((self._update_conversation_display, ()))

    def _handle_close(self):
        """Handle window close event (WM_DELETE_WINDOW)."""
        logger.info("Monitor window close button clicked.")
        # Option 1: Just close the monitor window
        # self.stop() # This would destroy the window

        # Option 2: Stop the entire application
        if self.app_controller and hasattr(self.app_controller, 'stop'):
            logger.info("Requesting application stop via controller.")
            # Call stop on the main app - this should eventually lead to monitor.stop() being called
            # Run stop in a separate thread to avoid blocking the UI thread if stop takes time
            threading.Thread(target=self.app_controller.stop, daemon=True).start()
        else:
            # Fallback if no controller, just stop the monitor itself
            logger.warning("No app controller reference found, stopping monitor only.")
            self.stop()

    def _refresh_thread(self):
        """Thread that refreshes the system status display"""
        while self.running:
            try:
                # Queue the status update instead of directly calling it from this thread
                self.update_queue.put((self._update_status_indicators, ()))
                
                # Sleep for the refresh interval
                time.sleep(self.refresh_rate)
                
            except Exception as e:
                logger.error(f"Error in refresh thread: {e}")
                
    def _update_status_indicators(self):
        """Update the status indicators for each component - MUST run on main thread"""
        if not self.running or not self.root or not self.root.winfo_exists():
            return
            
        try:
            for comp_name, component in self.components.items():
                if comp_name not in self.status_indicators:
                    continue
                    
                status_label = self.status_indicators[comp_name]
                if not status_label.winfo_exists():
                    continue
                
                # Get component status
                status = "Unknown"
                color = "gray"
                
                if hasattr(component, 'is_running'):
                    is_running = component.is_running
                    if callable(is_running):
                        is_running = is_running()
                        
                    if is_running:
                        status = "Running"
                        color = "green"
                    else:
                        status = "Stopped"
                        color = "red"
                        
                # Update the status label
                status_label.config(text=status, foreground=color)
                
        except Exception as e:
            logger.error(f"Error updating status indicators: {e}")
            
    def _update_conversation_display(self):
        """Update the conversation display - MUST run on main thread"""
        if not self.running or not self.root or not self.root.winfo_exists() or not self.conversation_display:
            return
            
        try:
            # Clear existing text
            self.conversation_display.config(state=tk.NORMAL)
            self.conversation_display.delete(1.0, tk.END)
            
            # Add history entries
            for entry in self.conversation_history:
                timestamp = entry['timestamp']
                speaker = entry['speaker_name']
                confidence = entry['confidence']
                transcript = entry['transcript']
                
                # Format the speaker line
                speaker_line = f"[{timestamp}] {speaker} ({confidence:.2f}): "
                self.conversation_display.insert(tk.END, speaker_line, "speaker")
                
                # Add the transcript text
                self.conversation_display.insert(tk.END, f"{transcript}\n\n", "transcript")
                
            # Scroll to the end
            self.conversation_display.see(tk.END)
            self.conversation_display.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Error updating conversation display: {e}")
            
    def run_ui(self):
        """Run the tkinter main loop - should be called from the main thread."""
        if not self.root or not self.root.winfo_exists():
             logger.error("Cannot run UI: Root window does not exist.")
             return

        if not self.running:
            # Start background tasks if not already started
            self.start()

        logger.info("Starting Tkinter main loop for monitor UI...")
        try:
            # Start the tkinter main loop - this blocks until the window is closed
            self.root.mainloop()
            logger.info("Tkinter main loop finished.")
        except Exception as e:
            logger.error(f"Error during Tkinter main loop: {e}", exc_info=True)
        finally:
            # Ensure running flag is set to false after mainloop exits
            self.running = False
            # We might not need to explicitly call stop here if _handle_close triggers app stop
            # self.stop()