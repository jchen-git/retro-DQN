from os import walk
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QSizePolicy, QSlider, QSpinBox, QStyleFactory, QTabWidget,
                             QTextEdit,
                             QVBoxLayout, QWidget, QDoubleSpinBox, QFileDialog, QMessageBox, QCheckBox)
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
# - Add feature to disable rendering
# - Add feature to create models with a specific name
# - Add feature to delete model
# - Remember to remove private data from hyperparameters.yaml

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
        training_tab.setFixedHeight(580)
        training_tab.setFixedWidth(560)
        self.output_box.setFixedHeight(250)
        self.output_box.setFixedWidth(390)
        self.output_box.setText("Ready")

        # Progress Bar
        self.training_prog_bar = QProgressBar()
        self.training_prog_bar.setFixedWidth(540)
        self.training_prog_bar.setRange(0, 300)
        self.training_prog_bar.setValue(0)

        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self.runTraining)
        self.run_model_button = QPushButton("Run Model")
        self.run_model_button.clicked.connect(self.runAgent)
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

        self.checkpoints = self.getCheckpoints()
        self.checkpointsComboBox = QComboBox()
        self.checkpointsComboBox.addItems(self.checkpoints.keys())

        checkpointsLabel = QLabel("Checkpoint:")
        checkpointsLabel.setBuddy(self.checkpointsComboBox)

        episode_slider_label = QLabel("Episodes:")
        self.episode_slider_input = QSpinBox()
        self.episode_slider_input.setRange(0, 9999)
        episode_slider = QSlider(Qt.Orientation.Horizontal)
        episode_slider.setRange(0, 9999)
        self.episode_slider_input.valueChanged.connect(episode_slider.setValue)
        episode_slider.valueChanged.connect(self.episode_slider_input.setValue)
        self.episode_slider_input.setValue(int(hyperparam["epoch"]))

        quick_setting_layout = QGridLayout()
        quick_setting_layout.addWidget(styleLabel, 0, 0)
        quick_setting_layout.addWidget(styleComboBox, 0, 1)
        quick_setting_layout.addWidget(checkpointsLabel, 1, 0)
        quick_setting_layout.addWidget(self.checkpointsComboBox, 1, 1)
        quick_setting_layout.addWidget(episode_slider_label, 2, 0)
        quick_setting_layout.addWidget(self.episode_slider_input, 2, 1)
        quick_setting_layout.addWidget(episode_slider, 3, 0, 1, 2)
        quick_setting_layout.setColumnStretch(1, 1)
        quick_setting_tab.setLayout(quick_setting_layout)

        # Top half of the GUI
        # Contains output box, run training, and run model buttons
        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_box)
        output_layout.addStretch(1)
        output_group_box.setLayout(output_layout)

        button_layout = QVBoxLayout()
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

        # Epsilon Decay Hyperparameter Slider and Double Spin Box
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

        hyperparam_grid_layout.addWidget(epsilon_decay_label, 5, 0, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.epsilon_decay_slider, 5, 1, Qt.AlignmentFlag.AlignTop)
        hyperparam_grid_layout.addWidget(self.epsilon_decay_input, 5, 2, Qt.AlignmentFlag.AlignTop)

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

    def runAgent(self):
        self.output_box.setText("Using model to run agent...")
        self.run_model_button.setText("Stop")
        self.run_model_button.clicked.disconnect()
        self.run_model_button.clicked.connect(self.kill_run_agent)
        self.p = Process(target=tetrisDQN_play.run, args=(self.episode_slider_input.value(),
                                                          self.checkpointsComboBox.currentText()))
        self.p.daemon = True
        self.p.start()

    def runTraining(self):
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
        self.p2 = Process(target=tetrisDQN_train.run, args=(self.render_checkbox.isChecked(),
                                                            self.checkpointsComboBox.currentText(),
                                                            self.training_episodes,
                                                            self.child_done_conn,
                                                            self.child_conn))
        self.p2.daemon = True
        self.p2.start()
        self.t1 = QTimer()
        self.t1.timeout.connect(self.training_listen)
        self.t1.start(100)
        self.t2 = QTimer()
        self.t2.timeout.connect(self.training_log_listen)
        self.t2.start(500)

    def training_listen(self):
        self.training_prog_bar.setValue(int(self.training_episodes.value) + 1)
        if self.parent_done_conn.poll():
            if self.parent_done_conn.recv():
                self.t1.stop()
                self.t2.stop()
                self.kill_training()

    def training_log_listen(self):
        if self.parent_conn.poll():
            self.output_box.append(self.parent_conn.recv())

    def getCheckpoints(self):
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
        self.run_model_button.clicked.connect(self.runAgent)
        self.p.terminate()
        self.p.join(timeout=1)
        self.output_box.setText("Ready")

    def kill_training(self):
        self.start_training_button.setText("Start Training")
        self.start_training_button.clicked.disconnect()
        self.start_training_button.clicked.connect(self.runTraining)
        self.p2.terminate()
        self.p2.join(timeout=1)
        self.output_box.setText("Ready")
        self.t1.stop()
        self.t2.stop()
        self.training_prog_bar.setValue(0)

    def quick_apply_settings(self):
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparam_sets = yaml.safe_load(file)

        all_hyperparam_sets['tetris']['epsilon_decay'] = self.epsilon_decay_input.value()
        all_hyperparam_sets['tetris']['epoch'] = self.episode_slider_input.value()
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

        all_hyperparam_sets['tetris']['epsilon_decay'] = self.epsilon_decay_input.value()
        all_hyperparam_sets['tetris']['epoch'] = self.episode_slider_input.value()
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

    def convert_epsilon_decay_slider(self, val):
        val = float(val/10000)
        self.epsilon_decay_input.setValue(val)

    def convert_epsilon_decay_dspin(self, val):
        val = int(val*10000)
        self.epsilon_decay_slider.setValue(val)

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

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        QApplication.setPalette(self.originalPalette)

    # Delete templates below when done with project
    # def advanceProgressBar(self):
    #     curVal = self.progressBar.value()
    #     maxVal = self.progressBar.maximum()
    #     self.progressBar.setValue(curVal + (maxVal - curVal) // 100)
    #
    # def createTopLeftGroupBox(self):
    #     self.topLeftGroupBox = QGroupBox("Group 1")
    #
    #     radioButton1 = QRadioButton("Radio button 1")
    #     radioButton2 = QRadioButton("Radio button 2")
    #     radioButton3 = QRadioButton("Radio button 3")
    #     radioButton1.setChecked(True)
    #
    #     checkBox = QCheckBox("Tri-state check box")
    #     checkBox.setTristate(True)
    #     checkBox.setCheckState(Qt.CheckState.PartiallyChecked)
    #
    #     layout = QVBoxLayout()
    #     layout.addWidget(radioButton1)
    #     layout.addWidget(radioButton2)
    #     layout.addWidget(radioButton3)
    #     layout.addWidget(checkBox)
    #     layout.addStretch(1)
    #     self.topLeftGroupBox.setLayout(layout)
    #
    # def createTopRightGroupBox(self):
    #     self.topRightGroupBox = QGroupBox("Group 2")
    #
    #     defaultPushButton = QPushButton("Change")
    #     defaultPushButton.setDefault(True)
    #
    #     togglePushButton = QPushButton("Toggle Push Button")
    #     togglePushButton.setCheckable(True)
    #     togglePushButton.setChecked(True)
    #
    #     flatPushButton = QPushButton("Flat Push Button")
    #     flatPushButton.setFlat(True)
    #
    #     layout = QVBoxLayout()
    #     layout.addWidget(defaultPushButton)
    #     layout.addWidget(togglePushButton)
    #     layout.addWidget(flatPushButton)
    #     layout.addStretch(1)
    #     self.topRightGroupBox.setLayout(layout)
    #
    # def createBottomRightGroupBox(self):
    #     self.bottomRightGroupBox = QGroupBox("Group 3")
    #     self.bottomRightGroupBox.setCheckable(True)
    #     self.bottomRightGroupBox.setChecked(True)
    #
    #     lineEdit = QLineEdit('s3cRe7')
    #
    #     spinBox = QDoubleSpinBox(self.bottomRightGroupBox)
    #     spinBox.setMinimum(-100)
    #     spinBox.setValue(-0.03)
    #
    #     dateTimeEdit = QDateTimeEdit(self.bottomRightGroupBox)
    #     dateTimeEdit.setDateTime(QDateTime.currentDateTime())
    #
    #     slider = QSlider(Qt.Orientation.Horizontal, self.bottomRightGroupBox)
    #     slider.setValue(40)
    #
    #     scrollBar = QScrollBar(Qt.Orientation.Horizontal, self.bottomRightGroupBox)
    #     scrollBar.setValue(60)
    #
    #     dial = QDial(self.bottomRightGroupBox)
    #     dial.setValue(30)
    #     dial.setNotchesVisible(True)
    #
    #     layout = QGridLayout()
    #     layout.addWidget(lineEdit, 0, 0, 1, 2)
    #     layout.addWidget(spinBox, 1, 0, 1, 2)
    #     layout.addWidget(dateTimeEdit, 2, 0, 1, 2)
    #     layout.addWidget(slider, 3, 0)
    #     layout.addWidget(scrollBar, 4, 0)
    #     layout.addWidget(dial, 3, 1, 2, 1)
    #     layout.setRowStretch(5, 1)
    #     self.bottomRightGroupBox.setLayout(layout)
    #
    # def createProgressBar(self):
    #     self.progressBar = QProgressBar()
    #     self.progressBar.setRange(0, 10000)
    #     self.progressBar.setValue(0)
    #
    #     timer = QTimer(self)
    #     timer.timeout.connect(self.advanceProgressBar)
    #     timer.start(1000)


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    interface = RetroDQNInterface()
    interface.show()
    sys.exit(app.exec())
