import struct
import numpy as np

def Hex2FP16(hexa: str):
    y = struct.pack("H", int(hexa, 16))
    float = np.frombuffer(y, dtype=np.float16)[0]
    print(float)  # 5.9
    return float

def getFP16Str(dec_float=5.9):
    # # 十进制单精度浮点转16位16进制
    hexa = struct.unpack('H', struct.pack('e', dec_float))[0]
    # print()
    hexa = hex(hexa)
    hexa = hexa[2:]

    # print(dec_float.tobytes())
    if len(hexa) != 4:
        hexa+=('0'*(4-len(hexa)))

    print(hexa)  # 45e6
    return hexa#str(dec_float.tobytes())#hexa

# Hex2FP16('3c00')
# Hex2FP16('4000')
# Hex2FP16('4700')
Hex2FP16('3be4')
Hex2FP16('2fd0')

getFP16Str(0.03)
getFP16Str(0.04)
getFP16Str(0.03*0.04)
print("here: ")
getFP16Str(2.5)
Hex2FP16('1400')
Hex2FP16('14ea')
print("here2: ")
Hex2FP16('c2fd')
Hex2FP16('2f23')

for i in range(6):
    print(i, end=' ')
    getFP16Str(i)

Hex2FP16('e334')
Hex2FP16("f1f8")
Hex2FP16("08c4")