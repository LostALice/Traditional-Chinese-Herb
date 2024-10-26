

from typing import Literal
import sqlite3


class Database(object):
    def __init__(self) -> None:
        self.connection = sqlite3.connect("inventory.db")
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
        self.init_database()

    def init_database(self) -> None:
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                quantity INTEGER NOT NULL DEFAULT 0
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                change_type TEXT NOT NULL,
                amount INTEGER NOT NULL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        """)

        self.connection.commit()

    def generate_stock_overview(self) -> list:
        self.cursor.execute("SELECT * FROM products")
        products = self.cursor.fetchall()

        return products

    def add_product(self, name: str, quantity: int) -> None:
        self.cursor.execute(
            "INSERT INTO products (name, quantity) VALUES (?, ?)", (name, quantity))

        self.connection.commit()

    def update_stock(self, product_id: int, amount: int, change_type: Literal["in", "out"] = "in") -> None:
        self.cursor.execute(
            "SELECT quantity FROM products WHERE id = ?", (product_id,))
        result = self.cursor.fetchone()

        print(result)
