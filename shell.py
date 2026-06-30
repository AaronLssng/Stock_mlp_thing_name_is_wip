import sys

from prompt_toolkit import prompt

from fetcher import main as man_fetch
from yfin import load_stock_from_binary
from mlp_prep import train_model

current_chosen_stock = None


# util funtions
def string_slicer(stri):
    start = 5
    end = len(stri)
    new_str = stri[:start] + stri[end + 1 :]
    return new_str


# Expressions in REPL
def open():
    def op_w_sys(stock):
        global current_chosen_stock
        try:
            current_chosen_stock = load_stock_from_binary(stock)
            print(current_chosen_stock.info())
            return
        except:
            print(f"Symbol {stock} in data!")
            print("Try again")
            return

    while True:
        print("Enter symbol")
        choice = input("> ")
        try:
            op_w_sys(choice)
            break
        except:
            print("invalid symbol")
            continue


def view(command):
    if current_chosen_stock is None:
        print("No stock opened with '.open'\n")
        return

    n_command = command.replace(".view", "")
    n_command = n_command.translate(str.maketrans("", "", "<>"))
    n_command = n_command.strip()

    if n_command == "" or n_command == "n":
        print(current_chosen_stock.tail(10))
        return

    try:
        casted_comand = int(n_command)
        print(current_chosen_stock.tail(casted_comand))
        return
    except ValueError:
        print(f"Could not cast: {n_command} ")
        return


def head():
    if current_chosen_stock is not None:
        print(current_chosen_stock.head())
        return
    else:
        print("No stock opened\n")
        return


def train(to_parse):




def take_input(txt):
    check_txt = string_slicer(txt)

    match check_txt:
        case ".open":
            return open()

        case ".view":
            return view(txt)

        case ".help":
            print(
                ".open : opens an bin file from data directory via Symbol\n"
                ".view <n> : gives the bars from bottom up of the opened stock\n"
                "'n' can be any number and represents the number of bars to view\n"
                "by defauflt 10 bars get shown. if you dont want to give 'n' you still have to write: '<n>'\n"
                "'.view only works if a stock has ben succesfully been opened with '.open'\n"
                "Example of valid calling of '.view': .view <5>\n"
            )
            ins = input("> ")
            return take_input(ins)

        case ".exit":
            print("Good Bye")
            return sys.exit(0)

        case ".head":
            return head()

        case ".fetc":
            return man_fetch()

        case _:
            print("Error occured")
            return main()


def main():
    while True:
        text = prompt("> ")
        take_input(text)


main()
