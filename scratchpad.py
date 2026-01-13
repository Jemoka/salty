from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("salty").setLevel(logging.DEBUG)


from reachy_mini import ReachyMini
from scipy.signal import resample
import time
from salty.realtime import RealtimeConfig, RealtimeHandler

with ReachyMini() as mini:
    # # Initialization - After this point, both audio devices (input/output) will be seen as busy by other applications!
    # mini.media.start_recording()
    # # mini.media.start_playing()

    # # Record
    # samples = None
    # while True:
    #     samples = mini.media.get_audio_sample()
    #     print(samples)

    # # Resample (if needed)
    # samples = resample(samples, mini.media.get_output_audio_samplerate()*len(samples)/mini.media.get_input_audio_samplerate())

    # # Play
    # mini.media.push_audio_sample(samples)
    # time.sleep(len(samples) / mini.media.get_output_audio_samplerate())

    # # Get Direction of Arrival
    # # 0 radians is left, π/2 radians is front/back, π radians is right.
    # doa, is_speech_detected = mini.media.get_DoA()
    # print(doa, is_speech_detected)

    # # Release audio devices (input/output)
    # mini.media.stop_recording()
    # mini.media.stop_playing()
