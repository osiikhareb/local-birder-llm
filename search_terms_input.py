# -*- coding: utf-8 -*-
"""
Optional Program to run to add additional search terms

@author: Osi
"""

import os
import csv


def get_user_input():
    return input("Enter a text string to store in the CSV: ").strip()

def csv_exists(filepath):
    return os.path.isfile(filepath)

def read_csv(filepath):
    with open(filepath, mode='r', newline='') as file:
        reader = csv.reader(file)
        return list(reader)

def write_to_csv(filepath, data):
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def user_wants_to_continue():
    while True:
        choice = input("Do you want to enter another input? (yes/no): ").strip().lower()
        if choice in ['yes', 'no']:
            return choice == 'yes'
        print("Invalid input. Please enter 'yes' or 'no'.")

def main():
    filepath = 'A:/Documents/Python Scripts/BirdBot3.0/search_terms.csv'
    
    if not csv_exists(filepath):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SearchTerm'])  # Writing header if the file is created
        print(f"CSV file '{filepath}' created with header.")

    existing_terms = set()
    if csv_exists(filepath):
        rows = read_csv(filepath)
        existing_terms.update(row[0] for row in rows[1:])  # Skip header

    while True:
        user_input = get_user_input()
        if user_input in existing_terms:
            print(f"The term '{user_input}' already exists in the CSV.")
        else:
            write_to_csv(filepath, [user_input])
            existing_terms.add(user_input)
            print(f"'{user_input}' has been added to the CSV.")

        if not user_wants_to_continue():
            break

if __name__ == "__main__":
    main()
