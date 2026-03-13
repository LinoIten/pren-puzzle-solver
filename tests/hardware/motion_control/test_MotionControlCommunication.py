import unittest
from unittest.mock import patch, MagicMock
import struct
import serial

from src.hardware.motion_control.MotionControlCommunication import MotionControlCommunicator
from src.utils.puzzle_piece import PuzzlePiece
from src.utils.Pose import Pose


class TestMotionControlCommunicator(unittest.TestCase):

    def setUp(self):
        """Set up a communicator instance for each test."""
        self.communicator = MotionControlCommunicator(serial_port='/dev/ttyFAKE', baudrate=115200)

    @patch('serial.Serial')
    def test_send_to_robot_with_valid_pieces(self, mock_serial_class):
        """
        Test that send_to_robot correctly formats and sends data for valid puzzle pieces.
        """
        # --- Arrange ---
        mock_serial_instance = MagicMock()
        mock_serial_class.return_value.__enter__.return_value = mock_serial_instance

        pieces = [
            PuzzlePiece(pid=1, pick=Pose(10, 20, 90)),
            PuzzlePiece(pid=2, pick=Pose(30, 40, 0))
        ]
        
        pieces[0].place_pose = Pose(100.0, 110.0, 180.0)
        pieces[1].place_pose = Pose(120.0, 130.0, -90.0)
            
        print(pieces)
        
        # --- Act ---
        self.communicator.send_to_robot(pieces)

        # --- Assert ---
        mock_serial_class.assert_called_once_with('/dev/ttyFAKE', 115200, timeout=2.0)

        expected_payload = bytearray()
        expected_payload.append(0x02)
        expected_payload.append(len(pieces))
        
        p1_id = 1
        p1_px, p1_py, p1_pt = 10.0, 20.0, 90.0
        p1_qx, p1_qy, p1_qt = 100.0, 110.0, 180.0
        p1_rot = (p1_qt - p1_pt) % 360.0 
        print(p1_rot)

        piece1_data = struct.pack('<Ifffff', p1_id, p1_px, p1_py, p1_qx, p1_qy, p1_rot)
        expected_payload.extend(piece1_data)

        p2_id = 2
        p2_px, p2_py, p2_pt = 30.0, 40.0, 0.0
        p2_qx, p2_qy, p2_qt = 120.0, 130.0, -90.0
        p2_rot = (p2_qt - p2_pt) % 360.0
        print(p2_rot)

        piece2_data = struct.pack('<Ifffff', p2_id, p2_px, p2_py, p2_qx, p2_qy, p2_rot)
        expected_payload.extend(piece2_data)

        checksum = 0
        for b in expected_payload[2:]:
            checksum ^= b
        expected_payload.append(checksum)

        mock_serial_instance.write.assert_called_once_with(expected_payload)
        mock_serial_instance.flush.assert_called_once()

    @patch('serial.Serial')
    def test_send_to_robot_empty_list(self, mock_serial_class):
        """
        Test that no data is sent for an empty list of pieces.
        """
        # --- Arrange ---
        mock_serial_instance = MagicMock()
        mock_serial_class.return_value.__enter__.return_value = mock_serial_instance

        # --- Act ---
        self.communicator.send_to_robot([])

        # --- Assert ---
        mock_serial_instance.write.assert_not_called()
        mock_serial_class.assert_not_called()


    @patch('builtins.print')
    @patch('serial.Serial')
    def test_serial_exception_handling(self, mock_serial_class, mock_print):
        """
        Test that a SerialException is caught and an error message is printed.
        """
        # --- Arrange ---
        error_message = "Permission denied"
        mock_serial_class.side_effect = serial.SerialException(error_message)

        pieces = [PuzzlePiece(pid=1, pick=Pose(10, 20, 0))]
        for p in pieces:
            p.place_pose = Pose(100, 110, 90)


        # --- Act ---
        self.communicator.send_to_robot(pieces)

        # --- Assert ---
        mock_serial_class.assert_called_once_with('/dev/ttyFAKE', 115200, timeout=2.0)
        
        mock_print.assert_any_call(f"Fehler bei der seriellen Kommunikation: {error_message}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

