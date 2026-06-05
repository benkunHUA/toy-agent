import os
from dotenv import load_dotenv
from src.query_engine import query_engine

load_dotenv()

def main():
  for chunk in query_engine("你好，现在是几点"):
    print(chunk, end="")


if __name__ == "__main__":
  main()
