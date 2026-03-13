import serial
import struct
from src.utils.puzzle_piece import PuzzlePiece

class MotionControlCommunicator:
    """
    Verwaltet die binäre serielle Kommunikation zwischen dem Raspberry Pi und dem Mikrocontroller.
    """
    
    def __init__(self, serial_port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.serial_port = serial_port
        self.baudrate = baudrate
        # Struktur-Definition: 
        # I = unsigned int (ID, 4 Bytes)
        # f = float (X, Y, X_Ziel, Y_Ziel, Rotation, je 4 Bytes)
        # Gesamt: 4 + (5 * 4) = 24 Bytes pro Teil
        self.piece_struct = struct.Struct('<Ifffff') 

    def send_to_robot(self, pieces: list[PuzzlePiece]) -> None:
        """
        Formatiert die Puzzleteile als binäres Paket und sendet sie.
        Format: [START_BYTE][ANZAHL][DATEN...][CHECKSUMME]
        """
        payload = bytearray()
        valid_pieces = [p for p in pieces if p.place_pose]

        if not valid_pieces:
            return

        payload.append(0x02) 
        payload.append(len(valid_pieces)) 

        for p in valid_pieces:
            px, py = p.pick_pose.x, p.pick_pose.y
            qx, qy = p.place_pose.x, p.place_pose.y
            rotation = (p.place_pose.theta - p.pick_pose.theta) % 360.0
            
            packed_piece = self.piece_struct.pack(
                int(p.id), float(px), float(py), float(qx), float(qy), float(rotation)
            )
            payload.extend(packed_piece)

        checksum = 0
        for b in payload[2:]:
            checksum ^= b
        payload.append(checksum)

        try:
            with serial.Serial(self.serial_port, self.baudrate, timeout=2.0) as ser:
                ser.write(payload)
                ser.flush()
                print(f"Binärdaten erfolgreich gesendet ({len(payload)} Bytes).")
        except serial.SerialException as e:
            print(f"Fehler bei der seriellen Kommunikation: {e}")