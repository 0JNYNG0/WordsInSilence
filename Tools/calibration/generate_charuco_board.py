"""
C0 — ChArUco 보드 PNG/PDF 생성기.

Usage:
    python Tools/calibration/generate_charuco_board.py

Output:
    Tools/calibration/charuco_board_7x5.png   (300 DPI, A4 landscape)
    Tools/calibration/charuco_board_7x5.pdf   (same image wrapped in PDF)

Requirements:
    pip install opencv-python  (opencv-contrib-python 도 OK)
"""

import json, os, sys
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(SCRIPT_DIR, "board_spec.json")

def main():
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        spec = json.load(f)

    sq_x = spec["squares_x"]        # 7
    sq_y = spec["squares_y"]        # 5
    sq_len = spec["square_length_mm"]   # 35.0
    mk_len = spec["marker_length_mm"]  # 26.0

    dict_name = spec["dictionary"]  # DICT_4X4_50
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    board = cv2.aruco.CharucoBoard(
        (sq_x, sq_y),
        sq_len / 1000.0,   # meters
        mk_len / 1000.0,
        aruco_dict,
    )

    # A4 landscape at 300 DPI: 297mm x 210mm -> 3508 x 2480 px
    # Board area: 245mm x 175mm
    # Add margins for centering
    dpi = 300
    a4_w_px = int(297 * dpi / 25.4)
    a4_h_px = int(210 * dpi / 25.4)

    board_w_mm = sq_x * sq_len  # 245
    board_h_mm = sq_y * sq_len  # 175
    board_w_px = int(board_w_mm * dpi / 25.4)
    board_h_px = int(board_h_mm * dpi / 25.4)

    # Generate board image (white background, black squares/markers)
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0)

    # Center on A4 canvas
    canvas = np.ones((a4_h_px, a4_w_px), dtype=np.uint8) * 255
    y_off = (a4_h_px - board_h_px) // 2
    x_off = (a4_w_px - board_w_px) // 2
    canvas[y_off:y_off + board_h_px, x_off:x_off + board_w_px] = board_img

    # Add corner crosshairs for print alignment verification
    ch_len = int(10 * dpi / 25.4)  # 10mm crosshair
    for cy, cx in [(y_off, x_off),
                   (y_off, x_off + board_w_px),
                   (y_off + board_h_px, x_off),
                   (y_off + board_h_px, x_off + board_w_px)]:
        cv2.line(canvas, (cx - ch_len, cy), (cx + ch_len, cy), 0, 2)
        cv2.line(canvas, (cx, cy - ch_len), (cx, cy + ch_len), 0, 2)

    # Add text label
    label = f"WordsInSilence ChArUco {sq_x}x{sq_y}  sq={sq_len}mm  mk={mk_len}mm  {dict_name}"
    font_scale = 1.2
    cv2.putText(canvas, label, (x_off, y_off - 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, 2, cv2.LINE_AA)

    # Save PNG
    png_path = os.path.join(SCRIPT_DIR, "charuco_board_7x5.png")
    cv2.imwrite(png_path, canvas)
    print(f"PNG saved: {png_path}  ({a4_w_px}x{a4_h_px} px, {dpi} DPI)")

    # Save PDF (simple: embed PNG via reportlab if available, else skip)
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.pdfgen import canvas as pdf_canvas
        from reportlab.lib.units import mm

        pdf_path = os.path.join(SCRIPT_DIR, "charuco_board_7x5.pdf")
        c = pdf_canvas.Canvas(pdf_path, pagesize=landscape(A4))
        pw, ph = landscape(A4)

        # Draw board image centered
        img_w = board_w_mm * mm
        img_h = board_h_mm * mm
        ix = (pw - img_w) / 2
        iy = (ph - img_h) / 2

        # Save board-only image temporarily for PDF embedding
        board_only_path = os.path.join(SCRIPT_DIR, "_board_only_temp.png")
        cv2.imwrite(board_only_path, board_img)
        c.drawImage(board_only_path, ix, iy, width=img_w, height=img_h)

        # Label
        c.setFont("Helvetica", 10)
        c.drawString(ix, iy + img_h + 10, label)

        c.save()
        os.remove(board_only_path)
        print(f"PDF saved: {pdf_path}")
    except ImportError:
        print("reportlab not installed -- PDF skipped. Install with: pip install reportlab")
        print("PNG can be printed directly at 100% scale on A4 landscape.")

    # Verification info
    print(f"\n=== Print Verification ===")
    print(f"Board grid: {sq_x} x {sq_y} squares")
    print(f"Expected board size: {board_w_mm} x {board_h_mm} mm")
    print(f"Square length: {sq_len} mm (MEASURE after printing!)")
    print(f"Marker length: {mk_len} mm (MEASURE after printing!)")
    print(f"Dictionary: {dict_name} (50 markers, 4x4 bit)")
    print(f"\nAfter printing, update board_spec.json 'measured' fields.")


if __name__ == "__main__":
    main()
