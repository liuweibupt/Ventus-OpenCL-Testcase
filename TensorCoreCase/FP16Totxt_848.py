import struct
import numpy as np
import os

dec_float = 5.9


def getFP16Str(dec_float=5.9):
    # # 十进制单精度浮点转16位16进制
    hexa = struct.unpack('H', struct.pack('e', dec_float))[0]
    # print()
    hexa = hex(hexa)
    hexa = hexa[2:]
    # print(hexa) # 45e6
    # print(dec_float.tobytes())
    return hexa#str(dec_float.tobytes())#hexa

# def getFP32Str(float_num):
#     # 将浮点数转换为4个字节的二进制数据
#     binary_data = struct.pack('f', float_num)
#     # 将二进制数据转换为十六进制字符串
#     hex_string = binary_data.hex()
#     return hex_string

def getFP32Str(float_value):
    # 将浮点数转换为4字节的二进制格式
    binary_data = struct.pack('!f', float_value)
    # 将二进制数据转换为十六进制字符串
    hex_string = binary_data.hex()
    return hex_string

def Hex2FP16(hexa: str):
    y = struct.pack("H", int(hexa, 16))
    float = np.frombuffer(y, dtype=np.float16)[0]
    print(float)  # 5.9
    return float


def MatFP16ToHexString(a):
    a_str = []
    for i in a:
        for j in i:
            a_str.append('h'+getFP16Str(j))
    return a_str

def MatFP32ToHexString(a):
    a_str = []
    for i in a:
        for j in i:
            a_str.append('h'+getFP32Str(j))
    return a_str

def MatABCD2Register(a, b, c, d, thread: int = 32, xLen = 32):
    Register_A = ['h' for i in range(thread)]
    Register_B = ['h' for i in range(thread)]
    Register_C = ['h' for i in range(thread)]
    Register_D = ['h' for i in range(thread)]
    for i in range(thread):
        if xLen ==64:
            Register_A[i] += (a[2 * i + 1 + 64] + a[2 * i + 64] + a[2 * i + 1] + a[2 * i])
            Register_B[i] += ('0000' + '0000' + b[2 * i + 1] + b[2 * i])
            Register_C[i] += (c[2 * i + 1 + 64] + c[2 * i + 64] + c[2 * i + 1] + c[2 * i])
            Register_D[i] += (d[2 * i + 1 + 64] + d[2 * i + 64] + d[2 * i + 1] + d[2 * i])
        elif xLen == 32:
            Register_A[i] += (a[2 * i + 1] + a[2 * i])
            Register_B[i] += (b[2 * i + 1] + b[2 * i])
            Register_C[i] += (c[2 * i + 1] + c[2 * i])
            Register_D[i] += (d[2 * i + 1] + d[2 * i])
    return Register_A, Register_B, Register_C, Register_D


def getMatABCD(m: int, n: int, k: int, result_path):
    a = np.random.randn(m, k).astype(np.float16)
    b = np.random.randn(k, n).astype(np.float16)
    c = np.random.randn(m, n).astype(np.float32)

    d = np.dot(a, b).astype(np.float16)
    d = (d+ c).astype(np.float32)

    print(d.dtype, d)
    np.savetxt(os.path.join(result_path, 'a.txt'), a)
    np.savetxt(os.path.join(result_path, 'b.txt'), b)
    np.savetxt(os.path.join(result_path, 'c.txt'), c)
    np.savetxt(os.path.join(result_path, 'd.txt'), d)

    print('Save Done!')
    return a, b, c, d


def strArr_Save(str_array, fname: str):
    # 使用空格将数组中的字符串连接起来
    space_separated_string = ' '.join(str_array)
    # 将连接后的字符串写入到TXT文件中
    with open(fname, 'w', encoding='utf-8') as file:
        file.write(space_separated_string)


if __name__ == '__main__':
    # print(getFP32Str(5.9))
    result_path = 'testData_848_mix'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if os.path.exists(os.path.join(result_path, 'a.txt')):
        a = np.loadtxt(os.path.join(result_path, 'a.txt'))
        b = np.loadtxt(os.path.join(result_path, 'b.txt'))
        c = np.loadtxt(os.path.join(result_path, 'c.txt'))
        # d = np.dot(a, b) + c
        # np.savetxt(os.path.join(result_path, 'd.txt'), d)
        d = np.loadtxt(os.path.join(result_path, 'd.txt'))
        print(np.dot(a,b))
    else:
        a, b, c, d = getMatABCD(8, 4, 8,result_path)
        print(getFP16Str(0))
        print("Get new data.")

    a_str = MatFP16ToHexString(a)
    b_str = MatFP16ToHexString(b.T)
    c_str = MatFP32ToHexString(c)
    d_str = MatFP32ToHexString(d)
    # print(a_str)
    # print(len(a_str))
    # RA, RB, RC, RD = MatABCD2Register(a_str, b_str, c_str, d_str)
    strArr_Save(a_str, os.path.join(result_path,'RA.txt'))
    strArr_Save(b_str, os.path.join(result_path,'RB.txt'))
    strArr_Save(c_str, os.path.join(result_path,'RC.txt'))
    strArr_Save(d_str, os.path.join(result_path,'RD.txt'))
    # strArr_Save(d_str, os.path.join(result_path,'RD_torch.txt'))
