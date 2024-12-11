from os import walk
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QSizePolicy, QSlider, QSpinBox, QStyleFactory, QTabWidget,
                             QTextEdit, QVBoxLayout, QWidget, QDoubleSpinBox, QFileDialog, QMessageBox, QCheckBox)
from multiprocessing import Process, Value, Pipe
import tetrisDQN_play
import tetrisDQN_train
import yaml

# TODO
# - More of what is shown in mockup
#   - Done
# - Connect buttons to scripts
#   - Done
# - Ability to save settings
#   - Done
# - Make training end properly instead of hanging on 99% and stop button still showing after completion
#   - Fixed by creating a new pipe and using it to send a done signal
# - Add more hyperparameters to the settings menu
#   - Done
# - Add feature to disable rendering
#   - Done
# - Add feature to create models with a specific name
#   - Done
# - Add feature to delete model
# - Remember to remove private data from hyperparameters.yaml
# - Test training using new empty .pt file
#   - Done
# - Test CUDA out of memory error
# - Change logging datetime
# - Separate output box and app state text
# - Implement setting to set hard stop to episode
#   - Fix UI for this

class RetroDQNInterface(QDialog):
    def __init__(self, parent=None):
        super(RetroDQNInterface, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        self.setFixedHeight(680)
        self.setFixedWidth(580)

        self.create_menu_tabs()

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.menu_tab_widget, 1, 0)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        self.setWindowTitle("Retro DQN")
        self.changeStyle('Windows11')

    def create_menu_tabs(self):
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparam_sets = yaml.safe_load(file)
            hyperparam = all_hyperparam_sets["tetris"]

        # Main menu
        self.menu_tab_widget = QTabWidget()
        self.menu_tab_widget.setSizePolicy(QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Ignored)
        output_group_box = QGroupBox("")
        buttons_group_box = QGroupBox("")

        self.menu_tab_widget.setFixedHeight(630)
        self.menu_tab_widget.setFixedWidth(560)
        training_layout = QHBoxLayout()
        training_tab = QWidget()
        self.output_box = QTextEdit()
        self.output_box.setText("Ready")

        # Progress Bar
        self.training_prog_bar = QProgressBar()
        self.training_prog_bar.setFixedWidth(540)
        self.training_prog_bar.setRange(0, 300)
        self.training_prog_bar.setValue(0)

        self.new_model_button = QPushButton("Create New Model")
        self.new_model_button.clicked.connect(self.new_model)
        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self.run_training)
        self.run_model_button = QPushButton("Run Model")
        self.run_model_button.clicked.connect(self.run_agent)
        self.render_checkbox = QCheckBox("Render Training")

        # Quick Settings tab on the Training menu tab
        quick_setting_widget = QTabWidget()
        quick_setting_tab = QWidget()
        quick_setting_widget.addTab(quick_setting_tab, "Quick Settings")

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        styleLabel = QLabel("Style:")
        styleLabel.setBuddy(styleComboBox)

        styleComboBox.textActivated.connect(self.changeStyle)

        self.models = self.get_models()
        self.models_combo_box = QComboBox()
        self.models_combo_box.addItems(self.models.keys())

        modelsLabel = QLabel("Checkpoint:")
        modelsLabel.setBuddy(self.models_combo_box)

        episode_slider_label = QLabel("Episodes:")
        self.episode_slider_input = QSpinBox()
        self.episode_slider_input.setRange(0, 9999)
        episode_slider = QSlider(Qt.Orientation.Horizontal)
        episode_slider.setRange(0, 9999)
        self.episode_slider_input.valueChanged.connect(episode_slider.setValue)
        episode_slider.valueChanged.connect(self.episode_slider_input.setValue)
        self.episode_slider_input.setValue(int(hyperparam['episodes']))

        self.episode_timer_toggle = QCheckBox("Use Episode Timer")
        episode_timer_label = QLabel("Episode Timer:")
        self.episode_timer_input = QSpinBox()
        self.episode_timer_input.setRange(0, 600)
        self.episode_timer_slider = QSlider(Qt.Orientation.Horizontal)
        self.episode_timer_slider.setRange(0, 600)
        self.episode_timer_input.valueChanged.connect(self.episode_timer_slider.setValue)
        self.episode_timer_slider.valueChanged.connect(self.episode_timer_input.setValue)
        self.episode_timer_toggle.clicked.connect(self.toggle_episode_timer)
        self.episode_timer_input.setValue(60)

        quick_setting_layout = QGridLayout()
        quick_setting_layout.addWidget(modelsLabel, 0, 0)
        quick_setting_layout.addWidget(self.models_combo_box, 0, 1)
        quick_setting_layout.addWidget(episode_slider_label, 1, 0)
        quick_setting_layout.addWidget(self.episode_slider_input, 1, 1)
        quick_setting_layout.addWidget(episode_slider, 2, 0, 1, 2)
        quick_setting_layout.addWidget(episode_timer_label, 3, 0)
        quick_setting_layout.addWidget(self.episode_timer_toggle, 3, 1)
        quick_setting_layout.addWidget(self.episode_timer_slider, 4, 0)
        quick_setting_layout.addWidget(self.episode_timer_input, 4, 1)
        quick_setting_layout.setColumnStretch(1, 1)
        quick_setting_tab.setLayout(quick_setting_layout)

        # Top half of the GUI
        # Contains output box, run training, and run model buttons
        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_box)
        output_layout.addStretch(1)
        output_group_box.setLayout(output_layout)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.new_model_button)
        button_layout.addWidget(self.start_training_button)
        button_layout.addWidget(self.run_model_button)
        button_layout.addWidget(self.render_checkbox)
        button_layout.addStretch(1)
        buttons_group_box.setLayout(button_layout)

        # Add the widgets to the main menu grid
        training_tab_grid = QGridLayout()
        training_tab_grid.addLayout(training_layout, 0, 0, 1, 2)
        training_tab_grid.addWidget(output_group_box, 1, 0)
        training_tab_grid.addWidget(buttons_group_box, 1, 1)
        training_tab_grid.addWidget(self.training_prog_bar, 2, 0, 1, 2)
        training_tab_grid.addWidget(quick_setting_widget, 3, 0, 1, 2)
        training_tab.setLayout(training_tab_grid)

        # Settings tab widgets
        apply_settings_group_box = QGroupBox("")
        apply_settings_group_box.setFixedHeight(50)
        apply_settings_group_box.setFixedWidth(536)
        apply_settings_layout = QVBoxLayout()
        apply_settings_button = QPushButton("Apply Settings")
        apply_settings_button.clicked.connect(self.apply_settings)
        apply_settings_layout.addWidget(apply_settings_button)
        apply_settings_group_box.setLayout(apply_settings_layout)

        hyperparam_group_box = QGroupBox("")
        hyperparam_grid_layout = QGridLayout()
        hyperparam_label = QLabel("Hyperparameters")
        hyperparam_val_label = QLabel("Value")

        # Reinforcement Learning Weights
        hole_weight_label = QLabel("Holes")
        self.hole_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.hole_weight_input = QDoubleSpinBox()
        self.hole_weight_slider.setRange(-10000, 10000)
        self.hole_weight_input.setRange(-1.0, 1.0)
        self.hole_weight_input.setSingleStep(0.0001)
        self.hole_weight_input.setDecimals(4)
        self.hole_weight_input.valueChanged['double'].connect(self.convert_hole_weight_dspin)
        self.hole_weight_slider.valueChanged['int'].connect(self.convert_hole_weight_slider)
        self.hole_weight_input.setValue(float(hyperparam['hole_weight']))

        agg_height_label = QLabel("Aggregate Height")
        self.agg_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.agg_height_input = QDoubleSpinBox()
        self.agg_height_slider.setRange(-10000, 10000)
        self.agg_height_input.setRange(-1.0, 1.0)
        self.agg_height_input.setSingleStep(0.0001)
        self.agg_height_input.setDecimals(4)
        self.agg_height_input.valueChanged['double'].connect(self.convert_agg_height_dspin)
        self.agg_height_slider.valueChanged['int'].connect(self.convert_agg_height_slider)
        self.agg_height_input.setValue(float(hyperparam['agg_height_weight']))

        bump_label = QLabel("Bumpiness")
        self.bump_slider = QSlider(Qt.Orientation.Horizontal)
        self.bump_input = QDoubleSpinBox()
        self.bump_slider.setRange(-10000, 10000)
        self.bump_input.setRange(-1.0, 1.0)
        self.bump_input.setSingleStep(0.0001)
        self.bump_input.setDecimals(4)
        self.bump_input.valueChanged['double'].connect(self.convert_bump_dspin)
        self.bump_slider.valueChanged['int'].connect(self.convert_bump_slider)
        self.bump_input.setValue(float(hyperparam['bumpiness_weight']))

        line_clear_label = QLabel("Line Clear")
        self.line_clear_slider = QSlider(Qt.Orientation.Horizontal)
        self.line_clear_input = QDoubleSpinBox()
        self.line_clear_slider.setRange(-10000, 10000)
        self.line_clear_input.setRange(-1.0, 1.0)
        self.line_clear_input.setSingleStep(0.0001)
        self.line_clear_input.setDecimals(4)
        self.line_clear_input.valueChanged['double'].connect(self.convert_line_clear_dspin)
        self.line_clear_slider.valueChanged['int'].connect(self.convert_line_clear_slider)
        self.line_clear_input.setValue(float(hyperparam['line_clear_weight']))

        # Initial epsilon value hyperparameter Slider and Double Spin Box
        epsilon_init_label = QLabel("Initial Epsilon")
        self.epsilon_init_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_init_input = QDoubleSpinBox()
        self.epsilon_init_slider.setRange(0, 10000)
        self.epsilon_init_input.setRange(0.0, 1.0)
        self.epsilon_init_input.setSingleStep(0.0001)
        self.epsilon_init_input.setDecimals(4)
        self.epsilon_init_input.setKeyboardTracking(False)
        self.epsilon_init_input.valueChanged['double'].connect(self.convert_epsilon_init_dspin)
        self.epsilon_init_slider.valueChanged['int'].connect(self.convert_epsilon_init_slider)
        self.epsilon_init_input.setValue(float(hyperparam['epsilon_init']))

        # Epsilon decay hyperparameter Slider and Double Spin Box
        epsilon_decay_label = QLabel("Epsilon Decay")
        self.epsilon_decay_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_decay_input = QDoubleSpinBox()
        self.epsilon_decay_slider.setRange(0, 10000)
        self.epsilon_decay_input.setRange(0.0, 1.0)
        self.epsilon_decay_input.setSingleStep(0.0001)
        self.epsilon_decay_input.setDecimals(4)
        self.epsilon_decay_input.setKeyboardTracking(False)
        self.epsilon_decay_input.valueChanged['double'].connect(self.convert_epsilon_decay_dspin)
        self.epsilon_decay_slider.valueChanged['int'].connect(self.convert_epsilon_decay_slider)
        self.epsilon_decay_input.setValue(float(hyperparam['epsilon_decay']))

        # Epsilon minimum hyperparameter Slider and Double Spin Box
        epsilon_min_label = QLabel("Epsilon Minimum")
        self.epsilon_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_min_input = QDoubleSpinBox()
        self.epsilon_min_slider.setRange(0, 10000)
        self.epsilon_min_input.setRange(0.0, 1.0)
        self.epsilon_min_input.setSingleStep(0.0001)
        self.epsilon_min_input.setDecimals(4)
        self.epsilon_min_input.setKeyboardTracking(False)
        self.epsilon_min_input.valueChanged['double'].connect(self.convert_epsilon_min_dspin)
        self.epsilon_min_slider.valueChanged['int'].connect(self.convert_epsilon_min_slider)
        self.epsilon_min_input.setValue(float(hyperparam['epsilon_min']))

        # Replay memory size hyperparameter Slider and Double Spin Box
        replay_size_label = QLabel("Replay Memory Size")
        self.replay_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.replay_size_input = QSpinBox()
        self.replay_size_slider.setRange(0, 100000)
        self.replay_size_input.setRange(0, 100000)
        self.replay_size_input.setSingleStep(1)
        self.replay_size_input.setKeyboardTracking(False)
        self.replay_size_input.valueChanged.connect(self.replay_size_slider.setValue)
        self.replay_size_slider.valueChanged.connect(self.replay_size_input.setValue)
        self.replay_size_input.setValue(int(hyperparam['replay_memory_size']))

        # Mini-batch size hyperparameter Slider and Double Spin Box
        mini_batch_size_label = QLabel("Mini-batch Size")
        self.mini_batch_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.mini_batch_size_input = QSpinBox()
        self.mini_batch_size_slider.setRange(0, 256)
        self.mini_batch_size_input.setRange(0, 256)
        self.mini_batch_size_input.setSingleStep(1)
        self.mini_batch_size_input.setKeyboardTracking(False)
        self.mini_batch_size_input.valueChanged.connect(self.mini_batch_size_slider.setValue)
        self.mini_batch_size_slider.valueChanged.connect(self.mini_batch_size_input.setValue)
        self.mini_batch_size_input.setValue(int(hyperparam['mini_batch_size']))

        # Learning rate hyperparameter Slider and Double Spin Box
        learning_rate_label = QLabel("Learning Rate")
        self.learning_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_slider.setRange(0, 10000)
        self.learning_rate_input.setRange(0.0, 1.0)
        self.learning_rate_input.setSingleStep(0.0001)
        self.learning_rate_input.setDecimals(4)
        self.learning_rate_input.setKeyboardTracking(False)
        self.learning_rate_input.valueChanged['double'].connect(self.convert_learning_rate_dspin)
        self.learning_rate_slider.valueChanged['int'].connect(self.convert_learning_rate_slider)
        self.learning_rate_input.setValue(float(hyperparam['learning_rate']))

        hyperparam_group_box.setLayout(hyperparam_grid_layout)

        hyperparam_grid_layout.addWidget(hyperparam_label, 0, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(hyperparam_val_label, 0, 1, Qt.AlignmentFlag.AlignTop)

        hyperparam_grid_layout.addWidget(hole_weight_label, 1, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.hole_weight_slider, 1, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.hole_weight_input, 1, 2, Qt.AlignmentFlag.AlignTop)

        hyperparam_grid_layout.addWidget(agg_height_label, 2, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.agg_height_slider, 2, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.agg_height_input, 2, 2, Qt.AlignmentFlag.AlignTop)

        hyperparam_grid_layout.addWidget(bump_label, 3, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.bump_slider, 3, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.bump_input, 3, 2, Qt.AlignmentFlag.AlignTop)

        hyperparam_grid_layout.addWidget(line_clear_label, 4, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.line_clear_slider, 4, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.line_clear_input, 4, 2, Qt.AlignmentFlag.AlignTop)

        hyperparam_grid_layout.addWidget(epsilon_init_label, 5, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.epsilon_init_slider, 5, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.epsilon_init_input, 5, 2, Qt.AlignmentFlag.AlignTop)

        hyperparam_grid_layout.addWidget(epsilon_decay_label, 6, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.epsilon_decay_slider, 6, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.epsilon_decay_input, 6, 2, Qt.AlignmentFlag.AlignTop)

        hyperparam_grid_layout.addWidget(epsilon_min_label, 7, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.epsilon_min_slider, 7, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.epsilon_min_input, 7, 2, Qt.AlignmentFlag.AlignTop)

        hyperparam_grid_layout.addWidget(replay_size_label, 8, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.replay_size_slider, 8, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.replay_size_input, 8, 2, Qt.AlignmentFlag.AlignTop)

        hyperparam_grid_layout.addWidget(mini_batch_size_label, 9, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.mini_batch_size_slider, 9, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.mini_batch_size_input, 9, 2, Qt.AlignmentFlag.AlignTop)

        hyperparam_grid_layout.addWidget(learning_rate_label, 10, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.learning_rate_slider, 10, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.learning_rate_input, 10, 2, Qt.AlignmentFlag.AlignTop)

        model_dir_group_box = QGroupBox("")
        model_dir_label = QLabel("Models Directory")
        model_dir_grid_layout = QGridLayout()
        self.model_dir_text_box = QLineEdit()
        self.model_dir_text_box.setFixedHeight(25)
        self.model_dir_text_box.setText(hyperparam['model_dir'])
        model_dir_browse_button = QPushButton("Browse...")
        model_dir_browse_button.clicked.connect(self.select_model_dir)
        model_dir_grid_layout.addWidget(model_dir_label, 0, 0)
        model_dir_grid_layout.addWidget(self.model_dir_text_box, 1, 0)
        model_dir_grid_layout.addWidget(model_dir_browse_button, 1, 1)
        model_dir_group_box.setLayout(model_dir_grid_layout)

        log_dir_group_box = QGroupBox("")
        log_dir_label = QLabel("Logs Directory")
        log_dir_grid_layout = QGridLayout()
        self.log_dir_text_box = QLineEdit()
        self.log_dir_text_box.setFixedHeight(25)
        self.log_dir_text_box.setText(hyperparam['log_dir'])
        log_dir_browse_button = QPushButton("Browse...")
        log_dir_browse_button.clicked.connect(self.select_log_dir)
        log_dir_grid_layout.addWidget(log_dir_label, 0, 0)
        log_dir_grid_layout.addWidget(self.log_dir_text_box, 1, 0)
        log_dir_grid_layout.addWidget(log_dir_browse_button, 1, 1)
        log_dir_group_box.setLayout(log_dir_grid_layout)

        settings_tab = QWidget()
        settings_tab_grid = QGridLayout()
        settings_tab_grid.addWidget(apply_settings_group_box, 0, 0)
        settings_tab_grid.addWidget(hyperparam_group_box, 1, 0)
        settings_tab_grid.addWidget(model_dir_group_box, 2, 0)
        settings_tab_grid.addWidget(log_dir_group_box, 3, 0)
        settings_tab.setLayout(settings_tab_grid)

        self.menu_tab_widget.addTab(training_tab, "Training")
        self.menu_tab_widget.addTab(settings_tab, "Settings")

    def run_agent(self):
        self.quick_apply_settings()
        self.output_box.setText("Using model to run agent...")
        self.run_model_button.setText("Stop")
        self.run_model_button.clicked.disconnect()
        self.run_model_button.clicked.connect(self.kill_run_agent)
        self.run_parent_done_conn, self.run_child_done_conn = Pipe()
        self.run_parent_conn, self.run_child_conn = Pipe()
        self.start_training_button.setDisabled(True)
        self.new_model_button.setDisabled(True)
        self.p = Process(target=tetrisDQN_play.run, args=(self.models_combo_box.currentText(),
                                                          self.model_dir_text_box.text(),
                                                          self.run_child_done_conn,
                                                          self.run_child_conn))
        self.p.daemon = True
        self.p.start()
        self.run_t1 = QTimer()
        self.run_t1.timeout.connect(self.run_model_listen)
        self.run_t1.start(100)

    def run_training(self):
        self.quick_apply_settings()
        self.training_prog_bar.setRange(0, int(self.episode_slider_input.value()))
        self.output_box.setText("Training in progress...")
        self.training_prog_bar.setValue(0)
        self.training_episodes = Value('i', 0)
        self.parent_conn, self.child_conn = Pipe()
        self.parent_done_conn, self.child_done_conn = Pipe()
        self.start_training_button.setText("Stop")
        self.start_training_button.clicked.disconnect()
        self.start_training_button.clicked.connect(self.kill_training)
        self.run_model_button.setDisabled(True)
        self.new_model_button.setDisabled(True)
        self.p2 = Process(target=tetrisDQN_train.run, args=(self.render_checkbox.isChecked(),
                                                            self.episode_timer_toggle.isChecked(),
                                                            self.models_combo_box.currentText(),
                                                            self.model_dir_text_box.text(),
                                                            self.log_dir_text_box.text(),
                                                            self.training_episodes,
                                                            self.child_done_conn,
                                                            self.child_conn,
                                                            self.episode_timer_input.value()))
        self.p2.daemon = True
        self.p2.start()
        self.t1 = QTimer()
        self.t1.timeout.connect(self.training_listen)
        self.t1.start(100)
        self.t2 = QTimer()
        self.t2.timeout.connect(self.training_log_listen)
        self.t2.start(500)

    def run_model_listen(self):
        if self.run_parent_done_conn.poll():
            run_state = self.run_parent_done_conn.recv()
            if run_state == 'done':
                self.run_t1.stop()
                self.kill_run_agent()

                if self.run_parent_conn.poll():
                    msg = QMessageBox()
                    msg.setWindowTitle('Score')
                    msg.setText(self.run_parent_conn.recv())
                    msg.exec()

            elif run_state == 'error':
                self.run_t1.stop()
                self.kill_run_agent()

                if self.run_parent_conn.poll():
                    msg = QMessageBox()
                    msg.setWindowTitle('Error')
                    msg.setText(self.run_parent_conn.recv())
                    msg.exec()

    def training_listen(self):
        self.training_prog_bar.setValue(int(self.training_episodes.value))
        if self.parent_done_conn.poll():
            if self.parent_done_conn.recv():
                self.t1.stop()
                self.t2.stop()
                self.kill_training()

    def training_log_listen(self):
        if self.parent_conn.poll():
            self.output_box.append(self.parent_conn.recv())

    def new_model(self):
        self.run_model_button.setDisabled(True)
        self.start_training_button.setDisabled(True)
        if self.model_dir_text_box.text() != '':
            filename, _ = QFileDialog.getSaveFileName(self, "Create New Model", self.model_dir_text_box.text() + "/model.pt")
        else:
            filename, _ = QFileDialog.getSaveFileName(self, "Create New Model", "model.pt")

        if filename != "":
            with open(filename, 'w') as file:
                file.write("")
            self.models_combo_box.addItem(filename.split('/')[-1])
            self.models_combo_box.setCurrentText(filename.split('/')[-1])
        self.run_model_button.setDisabled(False)
        self.start_training_button.setDisabled(False)

    def get_models(self):
        files_in_dir = next(walk("models"), (None, None, []))
        path = files_in_dir[0]
        ckpts = {}
        for file in files_in_dir[2]:
            if file[-3:] == ".pt":
                ckpts[file] = (path + f"/{file}")
        return ckpts

    def kill_run_agent(self):
        self.run_model_button.setText("Run Model")
        self.run_model_button.clicked.disconnect()
        self.run_model_button.clicked.connect(self.run_agent)
        self.p.terminate()
        self.p.join(timeout=1)
        self.output_box.setText("Ready")
        self.start_training_button.setDisabled(False)
        self.new_model_button.setDisabled(False)

    def kill_training(self):
        self.start_training_button.setText("Start Training")
        self.start_training_button.clicked.disconnect()
        self.start_training_button.clicked.connect(self.run_training)
        self.p2.terminate()
        self.p2.join(timeout=1)
        self.output_box.setText("Ready")
        self.t1.stop()
        self.t2.stop()
        self.training_prog_bar.setValue(0)
        self.run_model_button.setDisabled(False)
        self.new_model_button.setDisabled(False)

    def quick_apply_settings(self):
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparam_sets = yaml.safe_load(file)

        all_hyperparam_sets['tetris']['epsilon_init'] = self.epsilon_init_input.value()
        all_hyperparam_sets['tetris']['epsilon_decay'] = self.epsilon_decay_input.value()
        all_hyperparam_sets['tetris']['epsilon_min'] = self.epsilon_min_input.value()
        all_hyperparam_sets['tetris']['replay_memory_size'] = self.replay_size_input.value()
        all_hyperparam_sets['tetris']['mini_batch_size'] = self.mini_batch_size_input.value()
        all_hyperparam_sets['tetris']['learning_rate'] = self.learning_rate_input.value()
        all_hyperparam_sets['tetris']['episodes'] = self.episode_slider_input.value()
        all_hyperparam_sets['tetris']['bumpiness_weight'] = self.bump_input.value()
        all_hyperparam_sets['tetris']['agg_height_weight'] = self.agg_height_input.value()
        all_hyperparam_sets['tetris']['hole_weight'] = self.hole_weight_input.value()
        all_hyperparam_sets['tetris']['line_clear_weight'] = self.line_clear_input.value()
        all_hyperparam_sets['tetris']['model_dir'] = self.model_dir_text_box.text()
        all_hyperparam_sets['tetris']['log_dir'] = self.log_dir_text_box.text()

        with open('hyperparameters.yaml', 'w') as file:
            yaml.dump(all_hyperparam_sets, file, default_flow_style=False)

    def apply_settings(self):
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparam_sets = yaml.safe_load(file)

        all_hyperparam_sets['tetris']['epsilon_init'] = self.epsilon_init_input.value()
        all_hyperparam_sets['tetris']['epsilon_decay'] = self.epsilon_decay_input.value()
        all_hyperparam_sets['tetris']['epsilon_min'] = self.epsilon_min_input.value()
        all_hyperparam_sets['tetris']['learning_rate'] = self.learning_rate_input.value()
        all_hyperparam_sets['tetris']['replay_memory_size'] = self.replay_size_input.value()
        all_hyperparam_sets['tetris']['mini_batch_size'] = self.mini_batch_size_input.value()
        all_hyperparam_sets['tetris']['episodes'] = self.episode_slider_input.value()
        all_hyperparam_sets['tetris']['bumpiness_weight'] = self.bump_input.value()
        all_hyperparam_sets['tetris']['agg_height_weight'] = self.agg_height_input.value()
        all_hyperparam_sets['tetris']['hole_weight'] = self.hole_weight_input.value()
        all_hyperparam_sets['tetris']['line_clear_weight'] = self.line_clear_input.value()
        all_hyperparam_sets['tetris']['model_dir'] = self.model_dir_text_box.text()
        all_hyperparam_sets['tetris']['log_dir'] = self.log_dir_text_box.text()

        with open('hyperparameters.yaml', 'w') as file:
            yaml.dump(all_hyperparam_sets, file, default_flow_style=False)

        msg = QMessageBox()
        msg.setWindowTitle('Save')
        msg.setText('Settings saved!')
        msg.exec()

    def convert_epsilon_init_slider(self, val):
        val = float(val/10000)
        self.epsilon_init_input.setValue(val)

    def convert_epsilon_init_dspin(self, val):
        val = int(val*10000)
        self.epsilon_init_slider.setValue(val)

    def convert_epsilon_decay_slider(self, val):
        val = float(val/10000)
        self.epsilon_decay_input.setValue(val)

    def convert_epsilon_decay_dspin(self, val):
        val = int(val*10000)
        self.epsilon_decay_slider.setValue(val)

    def convert_epsilon_min_slider(self, val):
        val = float(val / 10000)
        self.epsilon_min_input.setValue(val)

    def convert_epsilon_min_dspin(self, val):
        val = int(val * 10000)
        self.epsilon_min_slider.setValue(val)

    def convert_learning_rate_slider(self, val):
        val = float(val / 10000)
        self.learning_rate_input.setValue(val)

    def convert_learning_rate_dspin(self, val):
        val = int(val * 10000)
        self.learning_rate_slider.setValue(val)

    def convert_hole_weight_slider(self, val):
        val = float(val/10000)
        self.hole_weight_input.setValue(val)

    def convert_hole_weight_dspin(self, val):
        val = int(val*10000)
        self.hole_weight_slider.setValue(val)

    def convert_agg_height_slider(self, val):
        val = float(val/10000)
        self.agg_height_input.setValue(val)

    def convert_agg_height_dspin(self, val):
        val = int(val*10000)
        self.agg_height_slider.setValue(val)

    def convert_bump_slider(self, val):
        val = float(val/10000)
        self.bump_input.setValue(val)

    def convert_bump_dspin(self, val):
        val = int(val*10000)
        self.bump_slider.setValue(val)

    def convert_line_clear_slider(self, val):
        val = float(val/10000)
        self.line_clear_input.setValue(val)

    def convert_line_clear_dspin(self, val):
        val = int(val*10000)
        self.line_clear_slider.setValue(val)

    def select_model_dir(self):
        file_dialog = QFileDialog()
        folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
        self.model_dir_text_box.setText(folder_path)

    def select_log_dir(self):
        file_dialog = QFileDialog()
        folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
        self.log_dir_text_box.setText(folder_path)

    def toggle_episode_timer(self):
        if self.episode_timer_toggle.isChecked():
            self.episode_timer_input.setDisabled(True)
            self.episode_timer_slider.setDisabled(True)
        else:
            self.episode_timer_input.setDisabled(False)
            self.episode_timer_slider.setDisabled(False)

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        QApplication.setPalette(self.originalPalette)

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    interface = RetroDQNInterface()
    interface.show()
    sys.exit(app.exec())
