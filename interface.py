from PyQt6.QtCore import QDateTime, Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QDoubleSpinBox)


class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        self.setFixedHeight(680)
        self.setFixedWidth(580)

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)

        self.useStylePaletteCheckBox = QCheckBox("&Use style's standard palette")
        self.useStylePaletteCheckBox.setChecked(False)

        disableWidgetsCheckBox = QCheckBox("&Disable widgets")

        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        self.create_menu_tabs()
        self.createBottomRightGroupBox()
        self.createProgressBar()

        styleComboBox.textActivated.connect(self.changeStyle)
        self.useStylePaletteCheckBox.toggled.connect(self.changePalette)
        disableWidgetsCheckBox.toggled.connect(self.topLeftGroupBox.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.topRightGroupBox.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.menu_tab_widget.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.bottomRightGroupBox.setDisabled)

        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)
        topLayout.addWidget(self.useStylePaletteCheckBox)
        topLayout.addWidget(disableWidgetsCheckBox)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.menu_tab_widget, 1, 0)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        self.setWindowTitle("Styles")
        self.changeStyle('Windows')

    def create_menu_tabs(self):
        self.menu_tab_widget = QTabWidget()
        self.menu_tab_widget.setSizePolicy(QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Ignored)
        output_group_box = QGroupBox("")
        buttons_group_box = QGroupBox("")

        self.menu_tab_widget.setFixedHeight(630)
        self.menu_tab_widget.setFixedWidth(560)
        training_layout = QHBoxLayout()
        training_tab = QWidget()
        output_box = QTextEdit()
        training_tab.setFixedHeight(580)
        training_tab.setFixedWidth(480)
        output_box.setFixedHeight(100)
        output_box.setFixedWidth(250)
        training_prog_bar = QProgressBar()
        training_prog_bar.setFixedWidth(680)
        training_prog_bar.setRange(0, 10000)
        training_prog_bar.setValue(0)

        start_training_button = QPushButton("Start Training")
        start_training_button.setDefault(True)
        run_model_button = QPushButton("Run Model")
        run_model_button.setDefault(True)

        output_layout = QVBoxLayout()
        output_layout.addWidget(output_box)
        output_layout.addStretch(1)
        output_group_box.setLayout(output_layout)

        button_layout = QVBoxLayout()
        button_layout.addWidget(start_training_button)
        button_layout.addWidget(run_model_button)
        button_layout.addStretch(1)
        buttons_group_box.setLayout(button_layout)

        training_tab_grid = QGridLayout()
        training_tab_grid.addLayout(training_layout, 0, 0, 1, 2)
        training_tab_grid.addWidget(output_group_box, 1, 0)
        training_tab_grid.addWidget(buttons_group_box, 1, 1)
        training_tab_grid.addWidget(training_prog_bar, 0, 0, 1, 2)
        training_tab.setLayout(training_tab_grid)

        settings_tab = QWidget()

        self.menu_tab_widget.addTab(training_tab, "Training")
        self.menu_tab_widget.addTab(settings_tab, "Settings")

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)

    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) // 100)

    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Group 1")

        radioButton1 = QRadioButton("Radio button 1")
        radioButton2 = QRadioButton("Radio button 2")
        radioButton3 = QRadioButton("Radio button 3")
        radioButton1.setChecked(True)

        checkBox = QCheckBox("Tri-state check box")
        checkBox.setTristate(True)
        checkBox.setCheckState(Qt.CheckState.PartiallyChecked)

        layout = QVBoxLayout()
        layout.addWidget(radioButton1)
        layout.addWidget(radioButton2)
        layout.addWidget(radioButton3)
        layout.addWidget(checkBox)
        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)

    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Group 2")

        defaultPushButton = QPushButton("Change")
        defaultPushButton.setDefault(True)

        togglePushButton = QPushButton("Toggle Push Button")
        togglePushButton.setCheckable(True)
        togglePushButton.setChecked(True)

        flatPushButton = QPushButton("Flat Push Button")
        flatPushButton.setFlat(True)

        layout = QVBoxLayout()
        layout.addWidget(defaultPushButton)
        layout.addWidget(togglePushButton)
        layout.addWidget(flatPushButton)
        layout.addStretch(1)
        self.topRightGroupBox.setLayout(layout)

    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("Group 3")
        self.bottomRightGroupBox.setCheckable(True)
        self.bottomRightGroupBox.setChecked(True)

        lineEdit = QLineEdit('s3cRe7')

        spinBox = QDoubleSpinBox(self.bottomRightGroupBox)
        spinBox.setMinimum(-100)
        spinBox.setValue(-0.03)

        dateTimeEdit = QDateTimeEdit(self.bottomRightGroupBox)
        dateTimeEdit.setDateTime(QDateTime.currentDateTime())

        slider = QSlider(Qt.Orientation.Horizontal, self.bottomRightGroupBox)
        slider.setValue(40)

        scrollBar = QScrollBar(Qt.Orientation.Horizontal, self.bottomRightGroupBox)
        scrollBar.setValue(60)

        dial = QDial(self.bottomRightGroupBox)
        dial.setValue(30)
        dial.setNotchesVisible(True)

        layout = QGridLayout()
        layout.addWidget(lineEdit, 0, 0, 1, 2)
        layout.addWidget(spinBox, 1, 0, 1, 2)
        layout.addWidget(dateTimeEdit, 2, 0, 1, 2)
        layout.addWidget(slider, 3, 0)
        layout.addWidget(scrollBar, 4, 0)
        layout.addWidget(dial, 3, 1, 2, 1)
        layout.setRowStretch(5, 1)
        self.bottomRightGroupBox.setLayout(layout)

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)
        #
        # timer = QTimer(self)
        # timer.timeout.connect(self.advanceProgressBar)
        # timer.start(1000)


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec())
