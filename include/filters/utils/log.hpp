#include <iostream>
#include <iomanip>

/**
 * @brief Log Colors for the terminal
 */
enum class LogColor
{
    RED,
    GREEN,
    YELLOW,
    BLUE,
    MAGENTA,
    CYAN,
    WHITE,
    RESET
};

/**
 * @brief Log function for debugging with different colors
 * @param msg Message to be printed
 * @param color Color of the message
 */
void vLog(int color, std::stringstream &ss)
{
  switch (color)
  {
    case 0: // RED
      std::cout << "\033[1;31m" << "\n" << ss.str() << "\033[0m";
      break;
    case 1: // GREEN
      std::cout << "\033[1;32m" << "\n" << ss.str() << "\033[0m";
      break;
    case 2: // YELLOW
      std::cout << "\033[1;33m" << "\n" << ss.str() << "\033[0m";
      break;
    case 3: // BLUE
      std::cout << "\033[1;34m" << "\n" << ss.str() << "\033[0m";
      break;
    case 4: // MAGENTA
      std::cout << "\033[1;35m" << "\n" << ss.str() << "\033[0m";
      break;
    case 5: // CYAN
      std::cout << "\033[1;36m" << "\n" << ss.str() << "\033[0m";
      break;
    case 6: // WHITE
      std::cout << "\033[1;37m" << "\n" << ss.str() << "\033[0m";
      break;
    default:
      std::cout << "\033[1;32m" << "\n" << ss.str() << "\033[0m";
      break;
  }
  ss.str("");
}
