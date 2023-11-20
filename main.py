import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model


# load_model: Hàm để tải mô hình học máy từ tệp h5.
# labels: Một từ điển ánh xạ từ số nguyên (được dự đoán từ mô hình) đến tên của loại trái cây hoặc rau củ.
# fruits: Một danh sách các loại trái cây.
# vegetables: Một danh sách các loại rau củ.
model = load_model('FV.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# xử lý hình ảnh đầu vào để đưa vào mô hình và trả về dự đoán được là loại trái cây hoặc rau củ.
def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()

# Hàm này dự đoán và hiển thị kết quả, loại trái cây/rau củ
def predict_and_display_result(img_path, label_var, category_var):
    result = processed_img(img_path)
    label_var.set("Predicted: " + result)

    if result in vegetables:
        category_var.set('Category: Vegetables')
    else:
        category_var.set('Category: Fruit')

# Hàm này mở hộp thoại để chọn một hình ảnh, sau đó hiển thị hình ảnh và kết quả dự đoán.
def open_file_dialog(label_var, category_var):
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg")])

    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(root, image=img)
        panel.image = img
        panel.grid(row=1, column=0, columnspan=3, pady=10)
        predict_and_display_result(file_path, label_var, category_var)


# Hàm này tạo cửa sổ giao diện tkinter và cấu hình các thành phần giao diện như nhãn, nút và các biến đối tượng tkinter.
# Sử dụng geometry để đặt kích thước và vị trí của cửa sổ.
# Gọi hàm mainloop() để bắt đầu vòng lặp chính của giao diện người dùng.
def run():
    global root
    root = tk.Tk()
    root.title("Nhận diện trái cây")
    window_width = 600
    window_height = 400
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = int((screen_width - window_width) / 2)
    y_coordinate = int((screen_height - window_height) / 2)
    root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    label_var = tk.StringVar()
    category_var = tk.StringVar()

    label_result = tk.Label(root, textvariable=label_var, font=("Helvetica", 14))
    label_result.grid(row=0, column=0, columnspan=3, pady=10)

    open_button = tk.Button(root, text="Open Image", command=lambda: open_file_dialog(label_var, category_var))
    open_button.grid(row=2, column=1, pady=10)

    category_label = tk.Label(root, textvariable=category_var, font=("Helvetica", 12))
    category_label.grid(row=3, column=0, columnspan=3, pady=5)

    root.mainloop()

run()
