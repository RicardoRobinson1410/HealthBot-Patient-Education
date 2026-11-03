"""
Main entry point for MediTech HealthBot.
Provides command-line execution and interactive patient education workflow.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bot import HealthBot


def main():
    """
    Main function to run the HealthBot interactive session.
    """
    try:
        bot = HealthBot()
        bot.run_interactive()
    except KeyboardInterrupt:
        print("\n\nSession terminated by patient. Stay healthy!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
