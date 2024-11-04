import os
import sys
import time
import pyperclip
import pyautogui
from audioplayer import AudioPlayer
from pynput.keyboard import Controller
from PyQt5.QtCore import QObject, QProcess
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QMessageBox

from key_listener import KeyListener
from result_thread import ResultThread
from ui.main_window import MainWindow
from ui.settings_window import SettingsWindow
from ui.status_window import StatusWindow
from transcription import create_local_model
from input_simulation import InputSimulator
from utils import ConfigManager

import ai_eval

class WhisperWriterApp(QObject):
    def __init__(self):
        """
        Initialize the application, opening settings window if no configuration file is found.
        """
        super().__init__()
        self.app = QApplication(sys.argv)
        self.app.setWindowIcon(QIcon(os.path.join('assets', 'ww-logo.png')))

        ConfigManager.initialize()

        self.settings_window = SettingsWindow()
        self.settings_window.settings_closed.connect(self.on_settings_closed)
        self.settings_window.settings_saved.connect(self.restart_app)

        self.activation_params = {}

        if ConfigManager.config_file_exists():
            self.initialize_components()
        else:
            print('No valid configuration file found. Opening settings window...')
            self.settings_window.show()

    def initialize_components(self):
        """
        Initialize the components of the application.
        """
        self.input_simulator = InputSimulator()
        self.key_listener = KeyListener()
        self.key_listener.add_callback("on_activate", self.on_activation)
        self.key_listener.add_callback("on_deactivate", self.on_deactivation)
        self.key_listener.add_callback("key_chord_activate", self.on_activation, {})
        self.key_listener.add_callback("copy_key_chord_activate", self.on_activation, {'copy':True, })
        self.key_listener.add_callback("eval_key_chord_activate", self.on_activation, {'eval':True, })
        self.key_listener.add_callback("eval_advanced_key_chord_activate", self.on_activation, {'advanced':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_key_chord_activate", self.on_activation, {'copy':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_advanced_key_chord_activate", self.on_activation, {'copy':True, 'advanced':True, 'eval':True, })
        self.key_listener.add_callback("eval_clipboard_key_chord_activate", self.on_activation, {'clipboard':True, 'eval':True, })
        self.key_listener.add_callback("eval_clipboard_advanced_key_chord_activate", self.on_activation, {'clipboard':True, 'advanced':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_clipboard_key_chord_activate", self.on_activation, {'copy':True, 'clipboard':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_clipboard_advanced_key_chord_activate", self.on_activation, {'copy':True, 'clipboard':True, 'advanced':True, 'eval':True, })
        self.key_listener.add_callback("eval_current_text_key_chord_activate", self.on_activation, {'current_text':True, 'eval':True, })
        self.key_listener.add_callback("eval_current_text_advanced_key_chord_activate", self.on_activation, {'current_text':True, 'advanced':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_current_text_key_chord_activate", self.on_activation, {'copy':True, 'current_text':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_current_text_advanced_key_chord_activate", self.on_activation, {'copy':True, 'current_text':True, 'advanced':True, 'eval':True, })
        self.key_listener.add_callback("key_chord_deactivate", self.on_deactivation, {})
        self.key_listener.add_callback("copy_key_chord_deactivate", self.on_deactivation, {'copy':True, })
        self.key_listener.add_callback("eval_key_chord_deactivate", self.on_deactivation, {'eval':True, })
        self.key_listener.add_callback("eval_advanced_key_chord_deactivate", self.on_deactivation, {'advanced':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_key_chord_deactivate", self.on_deactivation, {'copy':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_advanced_key_chord_deactivate", self.on_deactivation, {'copy':True, 'advanced':True, 'eval':True, })
        self.key_listener.add_callback("eval_clipboard_key_chord_deactivate", self.on_deactivation, {'clipboard':True, 'eval':True, })
        self.key_listener.add_callback("eval_clipboard_advanced_key_chord_deactivate", self.on_deactivation, {'clipboard':True, 'advanced':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_clipboard_key_chord_deactivate", self.on_deactivation, {'copy':True, 'clipboard':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_clipboard_advanced_key_chord_deactivate", self.on_deactivation, {'copy':True, 'clipboard':True, 'advanced':True, 'eval':True, })
        self.key_listener.add_callback("eval_current_text_key_chord_deactivate", self.on_deactivation, {'current_text':True, 'eval':True, })
        self.key_listener.add_callback("eval_current_text_advanced_key_chord_deactivate", self.on_deactivation, {'current_text':True, 'advanced':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_current_text_key_chord_deactivate", self.on_deactivation, {'copy':True, 'current_text':True, 'eval':True, })
        self.key_listener.add_callback("copy_eval_current_text_advanced_key_chord_deactivate", self.on_deactivation, {'copy':True, 'current_text':True, 'advanced':True, 'eval':True, })

        model_options = ConfigManager.get_config_section('model_options')
        model_path = model_options.get('local', {}).get('model_path')
        self.local_model = create_local_model() if not model_options.get('use_api') else None

        self.result_thread = None

        self.main_window = MainWindow()
        self.main_window.openSettings.connect(self.settings_window.show)
        self.main_window.startListening.connect(self.key_listener.start)
        self.main_window.closeApp.connect(self.exit_app)

        if not ConfigManager.get_config_value('misc', 'hide_status_window'):
            self.status_window = StatusWindow()

        self.create_tray_icon()
        self.main_window.show()

    def create_tray_icon(self):
        """
        Create the system tray icon and its context menu.
        """
        self.tray_icon = QSystemTrayIcon(QIcon(os.path.join('assets', 'ww-logo.png')), self.app)

        tray_menu = QMenu()

        show_action = QAction('WhisperWriter Main Menu', self.app)
        show_action.triggered.connect(self.main_window.show)
        tray_menu.addAction(show_action)

        settings_action = QAction('Open Settings', self.app)
        settings_action.triggered.connect(self.settings_window.show)
        tray_menu.addAction(settings_action)

        exit_action = QAction('Exit', self.app)
        exit_action.triggered.connect(self.exit_app)
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def cleanup(self):
        if self.key_listener:
            self.key_listener.stop()
        if self.input_simulator:
            self.input_simulator.cleanup()

    def exit_app(self):
        """
        Exit the application.
        """
        self.cleanup()
        QApplication.quit()

    def restart_app(self):
        """Restart the application to apply the new settings."""
        self.cleanup()
        QApplication.quit()
        QProcess.startDetached(sys.executable, sys.argv)

    def on_settings_closed(self):
        """
        If settings is closed without saving on first run, initialize the components with default values.
        """
        if not os.path.exists(os.path.join('src', 'config.yaml')):
            QMessageBox.information(
                self.settings_window,
                'Using Default Values',
                'Settings closed without saving. Default values are being used.'
            )
            self.initialize_components()

    def on_activation(self, copy = False, current_text = False, advanced = False, eval = False, clipboard = False):
        """
        Called when the activation key combination is pressed.
        """
        # print(f"Activation key pressed. Copy: {copy}, Current Text: {current_text}, Advanced: {advanced}, Eval: {eval}, Clipboard: {clipboard}")# Store activation parameters for later use

        if self.result_thread and self.result_thread.isRunning():
            recording_mode = ConfigManager.get_config_value('recording_options', 'recording_mode')
            if recording_mode == 'press_to_toggle':
                self.result_thread.stop_recording()
            elif recording_mode == 'continuous':
                self.stop_result_thread()
            elif recording_mode == 'voice_activity_detection':
                self.result_thread.stop_recording()
            return

        self.activation_params = {
            'copy': copy,
            'current_text': current_text,
            'advanced': advanced,
            'eval': eval,
            'clipboard': clipboard,
        }
        if current_text:
            # Artificially release all modifier keys
            pyautogui.keyUp('shift')
            pyautogui.keyUp('ctrl')
            pyautogui.keyUp('alt')
            pyautogui.keyUp('win')
            # Copy the currently selected text
            pyautogui.hotkey('ctrl', 'c')
        
        self.start_result_thread()

    def on_deactivation(self, copy = False, current_text = False, advanced = False, eval = False, clipboard = False):
        """
        Called when the activation key combination is released.
        """
        # print(f"Deactivation key pressed. Copy: {copy}, Current Text: {current_text}, Advanced: {advanced}, Eval: {eval}, Clipboard: {clipboard}")
        if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'hold_to_record':
            if self.result_thread and self.result_thread.isRunning():
                self.result_thread.stop_recording()

    def start_result_thread(self):
        """
        Start the result thread to record audio and transcribe it.
        """
        if self.result_thread and self.result_thread.isRunning():
            return

        self.result_thread = ResultThread(self.local_model)
        if not ConfigManager.get_config_value('misc', 'hide_status_window'):
            self.result_thread.statusSignal.connect(self.status_window.updateStatus)
            self.status_window.closeSignal.connect(self.stop_result_thread)
        self.result_thread.resultSignal.connect(self.on_transcription_complete)
        self.result_thread.start()

    def stop_result_thread(self):
        """
        Stop the result thread.
        """
        if self.result_thread and self.result_thread.isRunning():
            self.result_thread.stop()

    def on_transcription_complete(self, result):
        """
        When the transcription is complete, process the result and start listening for the activation key again.
        """
        self.handle_transcription_result(result)

        if ConfigManager.get_config_value('misc', 'noise_on_completion'):
            AudioPlayer(os.path.join('assets', 'beep.wav')).play(block=True)

        if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'continuous':
            self.start_result_thread()
        else:
            self.key_listener.start()

    def handle_transcription_result(self, transcription):
        """
        Handle the transcription based on activation parameters, to evaluate, copy or type the result.
        """
        # Decide if the transcription should be evaluated using LLMs
        if self.activation_params['eval']:
            if self.activation_params['current_text'] or self.activation_params['clipboard']:
                # Use the current clipboard content as context
                context = pyperclip.paste()
            else:
                context = None # No context

            # Evaluate the transcription
            print(f"Evaluating transcription: {transcription} with context: {context}{' on an advanced model' if self.activation_params['advanced'] else ''}")
            try:
                result = ai_eval.evaluate(
                    instructions=transcription,
                    context=context,
                    advanced=self.activation_params['advanced'],
                )
            except Exception as e:
                print(f"Error while evaluating: {e}")
                result = transcription
        else:
            result = transcription
        
        # Decide how to output the result
        if self.activation_params['copy']:
            # Copying to clipboard
            print(f"Copying to clipboard: {result}")
            pyperclip.copy(result)
        else:
            # Default behavior: Typing the result
            print(f"Typing result: {result}")
            self.custom_typewrite(result)

    def custom_typewrite(self, text):
        """
        Types the given text while properly handling line breaks.
        """
        # for char in text:
        #     self.input_simulator.typewrite(char)  # Type each character
        #     if char == '\n':
        #         self.input_simulator.typewrite('\n')  # Manually type a newline if needed
        #     # else:
        #         # # Optional: add a slight delay between characters
        #         # time.sleep(0.05)  # Adjust the delay as necessary
        if '\n' in text or len(text) > 20:
            pyperclip.copy(text)
            time.sleep(0.1)
            pyautogui.hotkey('ctrl', 'v')
        else:
            self.input_simulator.typewrite(text)

    def run(self):
        """
        Start the application.
        """
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    app = WhisperWriterApp()
    app.run()
