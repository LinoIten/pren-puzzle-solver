import serial
import struct
from src.utils.puzzle_piece import PuzzlePiece

class MotionControlCommunicator:
    """
    Verwaltet die binäre serielle Kommunikation gemäss dem Schnittstellen-Entwurf.
    """
    
    def __init__(self, serial_port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.serial_port = serial_port
        self.baudrate = baudrate
        # Struktur-Definition laut PDF[cite: 14, 17]:
        # Jeweils int32_t (4 Byte) für: Start X, Start Y, End X, End Y, Rotation
        # '<' erzwingt Little-Endian (LSB zuerst) 
        # '5i' steht für 5 signed 32-bit Integers (5 * 4 = 20 Bytes pro Teil)
        self.piece_struct = struct.Struct('<5i') 

    def send_to_robot(self, pieces: list[PuzzlePiece]) -> None:
        """
        Formatiert die Puzzleteile als binäres Paket gemäss PDF-Spezifikation.
        Format: [HEADER/FLAGS][PIECE 1...n][CHECKSUM] 
        """
        valid_pieces = [p for p in pieces if p.place_pose]
        if not valid_pieces:
            return

        payload = bytearray()

        header_flag = 0x01 if len(valid_pieces) == 6 else 0x00
        payload.append(header_flag)

        for p in valid_pieces:
            start_x = int(p.pick_pose.x)
            start_y = int(p.pick_pose.y)
            end_x = int(p.place_pose.x)
            end_y = int(p.place_pose.y)
            rotation = int((p.place_pose.theta - p.pick_pose.theta) % 360.0)
            
            packed_piece = self.piece_struct.pack(
                start_x, start_y, end_x, end_y, rotation
            )
            payload.extend(packed_piece)

        checksum = 0
        for b in payload:
            checksum ^= b
        payload.append(checksum)

        try:
            with serial.Serial(self.serial_port, self.baudrate, timeout=2.0) as ser:
                ser.write(payload)
                ser.flush()
                print(f"Binärdaten gemäss Entwurf gesendet ({len(payload)} Bytes).")
        except serial.SerialException as e:
            print(f"Fehler bei der seriellen Kommunikation: {e}")