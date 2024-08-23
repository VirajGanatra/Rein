#include <vector>
#include <stdexcept>
#include <random>

template <typename T>
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity)
            : capacity_(capacity), head_(0), tail_(0), size_(0), generator_(std::random_device{}()) {
        buffer_.resize(capacity);
    }

    void add(T item) {
        buffer_[tail_] = item;
        tail_ = (tail_ + 1) % capacity_;
        if (size_ < capacity_) {
            ++size_;
        } else {
            head_ = (head_ + 1) % capacity_;
        }
    }

    T sample() {
        if (isEmpty()) {
            throw std::runtime_error("Buffer is empty");
        }
        std::uniform_int_distribution<size_t> distribution(0, size_ - 1);
        size_t index = distribution(generator_);
        return buffer_[(head_ + index) % capacity_];
    }

    std::vector<T> random_batch(size_t batch_size) {
        if (batch_size > size_) {
            throw std::runtime_error("Batch size is greater than buffer size");
        }
        std::vector<T> batch;
        batch.reserve(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(sample());
        }
        return batch;
    }

    bool isEmpty() const {
        return size_ == 0;
    }

    size_t size() const {
        return size_;
    }

private:
    std::vector<T> buffer_;
    size_t capacity_;
    size_t head_;
    size_t tail_;
    size_t size_;
    std::mt19937 generator_;
};
