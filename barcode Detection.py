import os
import cv2
import numpy as np
from pyzbar.pyzbar import decode

print("cwd:", os.getcwd())
print("cv2:", cv2.__version__, "barcode module?", hasattr(cv2, "barcode"))

bardet = cv2.barcode.BarcodeDetector()

img_path = "BarcodeC128.png"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"cannot open image {img_path}")

def decode_with_fallback(img):
    # try OpenCV
    ok, info, typ, corners = bardet.detectAndDecodeWithType(img)
    if ok and any(s for s in info):
        # draw detected quads if present
        if corners is not None:
            for quad in corners:
                pts = quad.astype(int).reshape(-1, 2)
                cv2.polylines(img, [pts.reshape(-1,1,2)], True, (0,255,0), 2)
        return True, tuple(info), tuple(typ), img

    # fallback to pyzbar
    res = decode(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if not res:
        return False, (), (), img

    decoded = [r.data.decode("utf-8") for r in res]
    types = [r.type for r in res]
    for r in res:
        pts = np.array([(p.x, p.y) for p in r.polygon], np.int32).reshape(-1,1,2)
        cv2.polylines(img, [pts], True, (0,255,0), 2)
        cv2.putText(img, decoded[0], (r.rect.left, max(0, r.rect.top-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return True, tuple(decoded), tuple(types), img

# try original + four rotations; save first successful annotated image
for angle, flag in [(0, None), (90, cv2.ROTATE_90_CLOCKWISE),
                    (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]:
    test = img if flag is None else cv2.rotate(img, flag)
    ok, info, types, annotated = decode_with_fallback(test.copy())
    print(f"{angle:3d}° decode -> {ok} {info} {types}")
    if ok:
        out_name = f"decoded_{angle}.png"
        cv2.imwrite(out_name, annotated)
        print("Saved:", out_name)
        break