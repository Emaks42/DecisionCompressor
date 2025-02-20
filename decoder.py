from DTC import *
from PIL import Image


def convert_bytes_to_bits(data):
    bits = ""
    for byte in data:
        bits += f"{byte:08b}"
    return bits


def get_nodes_from_data(data, n: Node, bit_size_lim, arr_size_lim):
    not_end = False
    if data[:2] == "00":
        data = data[2:]
        n.answer = int(data[:bit_size_lim], 2)
        data = data[bit_size_lim:]
    elif data[:2] == "01":
        data = data[2:]
        n.feature = "x"
        n.threshold = int(data[:bit_size_lim], 2)
        data = data[bit_size_lim:]
        not_end = True
    elif data[:2] == "10":
        data = data[2:]
        n.feature = "y"
        n.threshold = int(data[:bit_size_lim], 2)
        data = data[bit_size_lim:]
        not_end = True
    elif data[:2] == "11":
        data = data[2:]
        count_ = int(data[:arr_size_lim], 2)
        data = data[arr_size_lim:]
        ans = []
        for obj in range(count_):
            ans.append(int(data[:bit_size_lim], 2))
            data = data[bit_size_lim:]
        n.answer = np.asarray(ans)
    if not_end:
        n.l, data = get_nodes_from_data(data, Node(), bit_size_lim, arr_size_lim)
        n.r, data = get_nodes_from_data(data, Node(), bit_size_lim, arr_size_lim)
    return n, data


input_ = "test_.dtcc"
file = open("compressed/"+input_, encoding="cp855")
code = bytes(file.read(), encoding="cp855")
width = int.from_bytes(code[:2])
height = int.from_bytes(code[2:4])
color_count = int.from_bytes(code[4:8])//2
color_size = 3 + int.from_bytes(code[4:8]) % 2
colors = []
for i in range(color_count):
    color = []
    for j in range(color_size):
        color.append(int.from_bytes(code[8 + i * color_size + j:8 + i * color_size + 1 + j]))
    colors.append(color)
bsl = max(len(bin(width)[2:]), len(bin(height)[2:]), len(bin(color_count)[2:]))
asl = int.from_bytes(code[8+color_size*color_count:8+color_size*color_count+4])
tree_data = code[8+color_size*color_count+4:]
tree = DecisionTree()
conv = convert_bytes_to_bits(tree_data)
tree.root, nul = get_nodes_from_data(conv, Node(), bsl, asl)
tree.print_conditions()
a = np.arange(0, height, 1)
b = np.arange(0, width, 1)
AB = np.meshgrid(a, b)
X = np.c_[AB[1].ravel(), AB[0].ravel()]
X = pd.DataFrame(data=X, columns=["x", "y"])
predict = tree.predict_(X, width, height)
picture_row = []
for i in predict.values:
    picture_row.append(colors[i])
picture = (np.asarray(picture_row)).reshape((width, height, color_size))
print(*picture)
