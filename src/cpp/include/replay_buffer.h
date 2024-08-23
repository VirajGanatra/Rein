#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include <vector>
#include <stdexcept>
#include <random>

template <typename T>
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity);

    void add(T item);
    T sample();
    std::vector<T> random_batch(size_t batch_size);
    bool isEmpty() const;
    size_t size() const;

private:
    std::vector<T> buffer_;
    size_t capacity_;
    size_t head_;
    size_t tail_;
    size_t size_;
    std::mt19937 generator_;
};

#include "replay_buffer.cpp"

#endif // REPLAY_BUFFER_H