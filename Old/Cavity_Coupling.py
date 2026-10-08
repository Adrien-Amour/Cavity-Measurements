from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QApplication
from PyQt5.QtCore import QTimer
from adriq.Counters import QuTau_Reader
from adriq.Servers import Client
import sys


class ChannelCountDisplay(QWidget):
    def __init__(self, count_reader):
        super().__init__()
        self.count_reader_client = Client(count_reader)

        # Set up main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Initialize max values
        self.channel3_max = 0
        self.channel4_max = 0

        # Add Channel 3 display
        self.add_channel_display("Channel 3", "green", "single_photon_chan3")  # Swapped to green

        # Add Channel 4 display
        self.add_channel_display("Channel 4", "blue", "single_photon_chan4")  # Swapped to blue

        # Timer to update the counts
        self.update_interval = 1000 // self.get_rate()  # Initial interval in ms
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_counts)
        self.timer.start(self.update_interval)

    def add_channel_display(self, channel_name, color, channel_key):
        """Add a display for a specific channel."""
        channel_layout = QHBoxLayout()

        # Channel label
        channel_label = QLabel(f"{channel_name}")
        channel_label.setStyleSheet(f"font-size: 24pt; color: {color};")
        channel_layout.addWidget(channel_label)

        # Current count box
        current_count_box = QLabel("N/A")
        current_count_box.setStyleSheet(
            "font-size: 24pt; background-color: white; border: 1px solid gray; border-radius: 5px; padding: 10px;"
        )
        current_count_box.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        current_count_box.setFixedWidth(200)  # Set a fixed width for the box
        channel_layout.addWidget(current_count_box)

        # Max count box
        max_count_box = QLabel("Max: N/A")
        max_count_box.setStyleSheet(
            "font-size: 18pt; background-color: white; border: 1px solid gray; border-radius: 5px; padding: 10px;"
        )
        max_count_box.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        max_count_box.setFixedWidth(200)  # Set a fixed width for the box
        channel_layout.addWidget(max_count_box)

        # Add layout to the main layout
        self.layout.addLayout(channel_layout)

        # Store references for updating
        setattr(self, f"{channel_key}_current_label", current_count_box)
        setattr(self, f"{channel_key}_max_label", max_count_box)
        setattr(self, f"{channel_key}_max", 0)

    def get_rate(self):
        """Retrieve the update rate from the count reader."""
        try:
            return self.count_reader_client.get_rate()
        except Exception as e:
            print(f"Error getting rate: {e}")
            return 10  # Default rate if unable to fetch

    def update_counts(self):
        """Fetch and update the most recent counts for all channels."""
        try:
            # Unpack the tuple returned by get_counts
            times, counts = self.count_reader_client.get_counts()

            # Update Channel 3
            self.update_channel_counts("single_photon_chan3", counts)

            # Update Channel 4
            self.update_channel_counts("single_photon_chan4", counts)

        except Exception as e:
            print(f"Error updating counts: {e}")

    def update_channel_counts(self, channel_key, counts):
        """Update the current and max counts for a specific channel."""
        current_label = getattr(self, f"{channel_key}_current_label")
        max_label = getattr(self, f"{channel_key}_max_label")
        max_value = getattr(self, f"{channel_key}_max")

        # Get the most recent count
        current_count = counts.get(channel_key, [0])[-1]  # Access the last value in the list
        current_label.setText(f"{current_count:.4g}")

        # Update max count if necessary
        if current_count > max_value:
            max_value = current_count
            setattr(self, f"{channel_key}_max", max_value)
        max_label.setText(f"Max: {max_value:.4g}")


if __name__ == "__main__":
    # Create the PyQt application
    app = QApplication(sys.argv)

    # Create and show the ChannelCountDisplay widget
    display = ChannelCountDisplay(QuTau_Reader)
    display.setWindowTitle("Channel Count Display")
    display.show()

    # Run the application event loop
    sys.exit(app.exec_())