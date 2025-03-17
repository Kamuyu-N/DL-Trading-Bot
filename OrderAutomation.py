import numpy as np
import pyautogui
import pytesseract
import time
from pywinauto import Application, timings


# Specify the Tesseract executable path if it's not in your PATH
# left = top_left_x
# top = top_left_y
# width = bottom_right_x - top_left_x
# height = bottom_right_y - top_left_y

left = 1365
top = 1114
width = 1446 - left
height = 1284 - top

sl, tp = 0.00077, 0.00087 #modify pip values for the stop loss and take profit

# Initialize
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
app = Application(backend="uia").connect(path='C:/Program Files/MetaTrader 5/terminal64.exe')
main_window = app.window(title_re=".*MetaQuotes.*")  # Adjust window title if needed

main_window.maximize()

main_window.wait('visible')  # Wait for the window to be ready

# Place the order
main_window.type_keys("{F9}")  # open the orders list
form_window = main_window.child_window(title="Order", control_type="Window")  # Adjust title

# Edit volume
volume = 0.6  # lot size
volume_box = form_window.child_window(title="Volume:", control_type="Edit")  # Adjust auto_id or use another property
volume_box.set_text(f"{volume}")  # Type text into the text box

# Set max deviation
deviation = 7
deviation_box = form_window.child_window(title='Deviation:', control_type='Edit')
deviation_box.set_text(f'{deviation}')

trade = 'buy'
if trade == 'buy':
    buy = form_window.child_window(title='Buy', control_type='Button')
    buy.click_input()
else:
    sell = form_window.child_window(title='Sell', control_type='Button')
    sell.click_input()

time.sleep(3)
region = (left, top, width, height)  # conatain all possible values
prices = pyautogui.screenshot(region=region).convert('L')
prices.save('test.png')

entry_prices = pytesseract.image_to_string(prices, config='--psm 6')

# create price list
price_list = []
for num in entry_prices.splitlines():
    try:
        price_list.append(float(num))
    except ValueError:
        continue

time.sleep(np.random.uniform(5.1, 12))  # Avoidd system Flagging

if len(price_list) > 1:
    trades_open = len(price_list)

    # adding the y value incase there are more than one orders
    top = top + 6  # amount to be selected during live markets

pyautogui.doubleClick(1366, 1138)  # Open modify window (Adjust According to window size )

modify = main_window.child_window(title_re='.*Position*.')

combo_box = main_window.child_window(auto_id="10338", control_type="ComboBox")
combo_box.expand()
dropdown_item = main_window.child_window(title="Modify Position", control_type="ListItem")
dropdown_item.click_input()
modify.print_control_identifiers()

# Modify s/l  and t/p levels
stop_loss = modify.child_window(title='Stop Loss:', control_type='Edit')
take_profit = modify.child_window(title='Take Profit:', auto_id="10336", control_type='Edit')

time.sleep(2)

if trade == 'buy':
    stop_loss.click_input()
    stop_loss.type_keys(f'{np.round(price_list[-1] - sl, 5)}')
    take_profit.click_input()
    take_profit.type_keys(f'{np.round(price_list[-1] + tp, 5)}')

else:
    stop_loss.click_input()
    stop_loss.type_keys(f'{np.round(price_list[-1] + sl, 5)}')
    take_profit.click_input()
    take_profit.type_keys(f'{np.round(price_list[-1] - tp, 5)}')

# Click the modify Button
modify_button = modify.child_window(title_re='.*Modify*.', control_type='Button')
modify_button.click_input()

