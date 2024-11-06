/**
 * @brief Circular buffer implementation which uses a std::array as the underlying container. It is templated on the type and size of the buffer.
 */

#pragma once

#include <array>
#include <algorithm>
#include <cstddef>
#include <iterator>


template<typename T, std::size_t SIZE>
class CircularBuffer
{
  public:
    /**
     * @brief Default constructor
     */
    CircularBuffer() : _buffer(SIZE),
                       _head(0), 
                       _tail(0), 
                       _size(0), 
                       _capacity(SIZE) { }

    /**
     * @brief The add function, this function adds the given element to the buffer and adjusts the head and tail if necessary
     * @param item element The element to add to the buffer
     */
    void add(const T& item)
    {
      // !< Insert the item at the head of the buffer
      _buffer[_head] = item;
      _head = (_head + 1) % _capacity;
      _size = std::min(_size + 1, _capacity);

      // !< If the buffer is full, increment the tail
      if(_full)
      {
        _tail = (_tail + 1) % _capacity;
      }
      else
      {
        _full = _head == _tail;
      }
    }

    /**
     * @brief The get function, this function returns the element at the given index
     * @param index The index of the element to return
     * @return The element at the given index
     */
    T get(std::size_t index) const
    {
      // !< If the index is out of bounds, throw an exception
      if(index >= _size)
      {
        throw std::out_of_range("Index out of range");
      }

      // !< Return the element at the given index
      return _buffer[(_tail + index) % _capacity];
    }

    /**
     * @brief The set function, this function sets the element at the given index to the given value
     * @param index The index of the element to set
     * @param item The value to set the element to
     */
    void set(std::size_t index, const T& item)
    {
      // !< If the index is out of bounds, throw an exception
      if(index >= _size)
      {
        throw std::out_of_range("Index out of range");
      }
      // !< Return the element at the given index
     
      _buffer[(_tail + index) % _capacity] = item;
    }

    /**
     * @brief Removes the last n elements from the buffer by moving the head back n elements
     * @param n The number of elements to remove
     */
    void removeHead(std::size_t n = 1)
    {
      if (n == 0)
      {
        return;
      }
      // !< If the number of elements to remove is greater than the size of the buffer, throw an exception
      if(n > _size)
      {
        throw std::out_of_range("Cannot remove more elements than the size of the buffer");
      }

      // !< Move the head back n elements
      _head = (_head - n) % _capacity;
      _size = std::max(_size - n, 0UL);
      _full = false;
    }

    /**
     * @brief Removes the first n elements from the buffer by moving the tail forward n elements
     * @param n The number of elements to remove
     */
    void removeTail(std::size_t n = 1)
    {
      if (n == 0)
      {
        return;
      }
      // !< If the number of elements to remove is greater than the size of the buffer, throw an exception
      if(n > _size)
      {
        throw std::out_of_range("Cannot remove more elements than the size of the buffer");
      }

      // !< Move the tail forward n elements
      _tail = n ? (_tail + n) % _capacity : 0;
      _size = n ? std::max(_size - n, 0UL) : 0;
      _full = false;
    }

    /**
     * @brief The size function, this function returns the size of the buffer
     * @return The size of the buffer
     */
    std::size_t size() const { return _size; }
    
    /**
     * @brief The capacity function, this function returns the capacity of the buffer
     * @return The capacity of the buffer
     */
    std::size_t capacity() const { return SIZE; }

    /**
     * @brief The empty function, this function returns true if the buffer is empty
     * @return True if the buffer is empty
     */
    bool empty() const { return !_full && _head == _tail; }

    /**
     * @brief The full function, this function returns true if the buffer is full
     * @return True if the buffer is full
     */
    bool full() const { return _full; }

  private:
    std::vector<T> _buffer; // !< The underlying buffer
                                 
    std::size_t _head;           // !< The index of the head of the buffer
    std::size_t _tail;           // !< The index of the tail of the buffer
    std::size_t _size;           // !< The number of elements in the buffer
    std::size_t _capacity;       // !< The capacity of the buffer
                                
    bool _full;                  // !< Flag that indicates if the buffer is full

};
