# 十六进制
def ascii_to_hex(data):
    data_hex = ""
    for i in range(len(data)):
        data_hex = data_hex + hex(ord(data[i]))[2:].upper() + " "  # 字符->10进制ascii->16进制 去掉0x
    return data_hex


def hex_to_ascii(data):
    ascii = ""
    hex_list = data.split(" ")
    count=hex_list.count('')#统计''出现次数
    for i in range(count):
        hex_list.remove('')# 删除''

    if len(hex_list) != 0:
        if hex_list[-1]!=' ':
            hex_list.append(' ')
            hex_list.pop(-1)
    else:
        pass
    for i in hex_list:
        hex = '0x' + i
        ascii += chr(int(hex, 16))  # 16进制->10进制ascii->字符
    return ascii


if __name__ == '__main__':
    s = input("请输入：")
    a = ascii_to_hex(s)
    print(a)

    b = hex_to_ascii(a)
    print(b)
