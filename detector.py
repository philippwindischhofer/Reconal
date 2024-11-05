import json, defs

class Detector:

    def __init__(self, json_path, var = "nominal"):
        with open(json_path, 'r') as infile:
            self.data = json.load(infile)[var]

    def get_channel_positions(self, station_id, channels):
        channel_positions = {}

        for channel in channels:
            for entry_id, entry in self.data["channels"].items():
                if entry["station_id"] == station_id and entry["channel_id"] == channel:
                    channel_positions[channel] = [entry["ant_position_x"] / defs.cvac, entry["ant_position_y"] / defs.cvac, entry["ant_position_z"] / defs.cvac]

        return channel_positions

    def get_cable_delays(self, station_id, channels):
        cable_delays = {}

        for channel in channels:
            for entry_id, entry in self.data["channels"].items():
                if entry["station_id"] == station_id and entry["channel_id"] == channel:
                    cable_delays[channel] = entry["cab_time_delay"]

        return cable_delays
