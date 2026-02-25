import serial
from src.utils.puzzle_piece import PuzzlePiece

class MotionControlCommunicator:
    """
    Verwaltet die serielle Kommunikation zwischen dem Raspberry Pi und dem Mikrocontroller.
    """
    
    def __init__(self, serial_port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.serial_port = serial_port
        self.baudrate = baudrate

    def send_to_robot(self, pieces: list[PuzzlePiece]) -> None:
        """
        Formatiert die geloesten Puzzleteile nach dem definierten ASCII-Protokoll
        Format: [<ID,START_X,START_Y,ZIEL_X,ZIEL_Y,ROTATION>;...]
        und sendet sie ueber die serielle Schnittstelle an den Mikrocontroller.
        Die Werte werden mit maximaler Genauigkeit (ohne Rundung) uebertragen.
        """
        piece_strings = []
        
        for p in pieces:
            if not p.place_pose:
                continue 
                
            px, py = p.pick_pose.x, p.pick_pose.y
            qx, qy = p.place_pose.x, p.place_pose.y
            
            rotation = (p.place_pose.theta - p.pick_pose.theta) % 360.0
            
            piece_str = f"<{p.id},{px},{py},{qx},{qy},{rotation}>"
            piece_strings.append(piece_str)
            
        if not piece_strings:
            return

        payload = "[" + ";".join(piece_strings) + "]\n"
        
        try:
            with serial.Serial(self.serial_port, self.baudrate, timeout=2.0) as ser:
                ser.write(payload.encode('ascii'))
                print("Daten erfolgreich gesendet.")
        except serial.SerialException as e:
            print(f"Fehler bei der seriellen Kommunikation: {e}")