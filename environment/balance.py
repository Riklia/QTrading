from pathlib import Path
import datetime


class Balance:
    def __init__(self, record_history: str | None = None):
        self._balance = dict()
        if record_history:
            self.record_history = Path(record_history)
            self.record_history.parent.mkdir(parents=True, exist_ok=True)
            with open(self.record_history, "w"):
                pass
        else:
            self.record_history = None

    def __getitem__(self, key: str):
        # the currency not in the balance dict => its balance 0
        return float(self._balance.get(key, 0))

    def __setitem__(self, key: str, value: float):
        raise ValueError("Assignment not allowed")

    def __str__(self):
        if not self._balance:
            return "Balance is 0.\n"

        output = "Balance:\n"
        for key, value in self._balance.items():
            output += f"{key}: {value:.8f}\n"
        return output

    def register_currency(self, currency_name: str):
        if currency_name not in self._balance.keys():
            self._balance[currency_name] = 0

    def update_balance(self, currency_name: str, delta: float):
        # for now, the unregistered currency is registered in this function, but
        # maybe in the future it can be handled in another way

        # also, the concept of "time" should be reconsidered - we are writing "balance history",
        # not the program logs, so should environment time be used? Maybe use singleton for time?
        # In this case this singleton time can be used in the environment and the balance classes
        # On the other side, Balance can be used just as dictionary, and all trading logic will be implemented in
        # environment - what makes sense, because environment is about "what happens", "what is the state".
        if currency_name not in self._balance.keys():
            self.register_currency(currency_name)

        self._balance[currency_name] += delta
        if self.record_history:
            ts = datetime.datetime.now().isoformat(timespec='seconds')
            log_message = ts + f" {currency_name}: "
            if delta >= 0:
                log_message += "+"
            log_message += str(delta) + f" | {self._balance[currency_name]}\n"
            with open(self.record_history, "a") as file:
                file.write(log_message)

