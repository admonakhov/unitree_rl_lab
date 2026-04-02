#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>

#include <errno.h>
#include <fcntl.h>
#include <linux/joystick.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

namespace isaaclab
{

class Gamepad
{
public:
    explicit Gamepad(std::string device = "/dev/input/js0")
    : device_(std::move(device))
    {
        open_device();
    }

    ~Gamepad()
    {
        close_device();
    }

    bool connected() const { return fd_ >= 0; }

    const unitree::common::UnitreeJoystick & joystick() const { return joystick_; }

    void apply_to(unitree::common::UnitreeJoystick & dst) const
    {
        dst.back(joystick_.back());
        dst.start(joystick_.start());
        dst.LS(joystick_.LS());
        dst.RS(joystick_.RS());
        dst.LB(joystick_.LB());
        dst.RB(joystick_.RB());
        dst.A(joystick_.A());
        dst.B(joystick_.B());
        dst.X(joystick_.X());
        dst.Y(joystick_.Y());
        dst.up(joystick_.up());
        dst.down(joystick_.down());
        dst.left(joystick_.left());
        dst.right(joystick_.right());
        dst.F1(joystick_.F1());
        dst.F2(joystick_.F2());
        dst.lx(joystick_.lx());
        dst.ly(joystick_.ly());
        dst.rx(joystick_.rx());
        dst.ry(joystick_.ry());
        dst.LT(joystick_.LT());
        dst.RT(joystick_.RT());
    }

    void update()
    {
        if (!ensure_open()) {
            return;
        }
        read_events();
        apply_to_joystick();
    }

private:
    void open_device()
    {
        if (fd_ >= 0) {
            return;
        }
        fd_ = ::open(device_.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd_ < 0) {
            last_open_attempt_ = std::chrono::steady_clock::now();
            return;
        }
        spdlog::info("Gamepad connected: {}", device_);
    }

    void close_device()
    {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool ensure_open()
    {
        if (fd_ >= 0) {
            return true;
        }
        auto now = std::chrono::steady_clock::now();
        if (now - last_open_attempt_ < std::chrono::seconds(1)) {
            return false;
        }
        open_device();
        return fd_ >= 0;
    }

    void read_events()
    {
        js_event e;
        while (true) {
            const ssize_t n = ::read(fd_, &e, sizeof(e));
            if (n == static_cast<ssize_t>(sizeof(e))) {
                const uint8_t type = e.type & ~JS_EVENT_INIT;
                if (type == JS_EVENT_AXIS) {
                    if (e.number < axes_.size()) {
                        axes_[e.number] = e.value;
                    }
                } else if (type == JS_EVENT_BUTTON) {
                    if (e.number < buttons_.size()) {
                        buttons_[e.number] = (e.value != 0) ? 1 : 0;
                    }
                }
                continue;
            }
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                break;
            }
            if (n < 0 && (errno == ENODEV || errno == EBADF)) {
                spdlog::warn("Gamepad disconnected: {}", device_);
                close_device();
            }
            break;
        }
    }

    static float norm_axis(int16_t v)
    {
        constexpr float denom = 32767.0f;
        float f = static_cast<float>(v) / denom;
        return std::clamp(f, -1.0f, 1.0f);
    }

    static float norm_trigger(int16_t v)
    {
        if (v >= 0 && v <= 255) {
            return static_cast<float>(v) / 255.0f;
        }
        float f = (static_cast<float>(v) + 32767.0f) / 65534.0f;
        return std::clamp(f, 0.0f, 1.0f);
    }

    void apply_to_joystick()
    {
        const float dpad_x = (axes_.size() > 6) ? norm_axis(axes_[6]) : 0.0f;
        const float dpad_y = (axes_.size() > 7) ? norm_axis(axes_[7]) : 0.0f;
        const bool dpad_left = (dpad_x < -0.5f) || (buttons_.size() > 13 && buttons_[13]);
        const bool dpad_right = (dpad_x > 0.5f) || (buttons_.size() > 14 && buttons_[14]);
        const bool dpad_up = (dpad_y < -0.5f) || (buttons_.size() > 11 && buttons_[11]);
        const bool dpad_down = (dpad_y > 0.5f) || (buttons_.size() > 12 && buttons_[12]);

        joystick_.A(buttons_.size() > 0 ? buttons_[0] : 0);
        joystick_.B(buttons_.size() > 1 ? buttons_[1] : 0);
        joystick_.X(buttons_.size() > 2 ? buttons_[2] : 0);
        joystick_.Y(buttons_.size() > 3 ? buttons_[3] : 0);
        joystick_.LB(buttons_.size() > 4 ? buttons_[4] : 0);
        joystick_.RB(buttons_.size() > 5 ? buttons_[5] : 0);
        joystick_.back(buttons_.size() > 6 ? buttons_[6] : 0);
        joystick_.start(buttons_.size() > 7 ? buttons_[7] : 0);
        joystick_.LS(buttons_.size() > 9 ? buttons_[9] : 0);
        joystick_.RS(buttons_.size() > 10 ? buttons_[10] : 0);

        joystick_.up(dpad_up ? 1 : 0);
        joystick_.down(dpad_down ? 1 : 0);
        joystick_.left(dpad_left ? 1 : 0);
        joystick_.right(dpad_right ? 1 : 0);

        joystick_.lx(axes_.size() > 0 ? norm_axis(axes_[0]) : 0.0f);
        joystick_.ly(axes_.size() > 1 ? -norm_axis(axes_[1]) : 0.0f);
        joystick_.rx(axes_.size() > 3 ? norm_axis(axes_[3]) : 0.0f);
        joystick_.ry(axes_.size() > 4 ? -norm_axis(axes_[4]) : 0.0f);
        joystick_.LT(axes_.size() > 2 ? norm_trigger(axes_[2]) : 0.0f);
        joystick_.RT(axes_.size() > 5 ? norm_trigger(axes_[5]) : 0.0f);
    }

    int fd_{-1};
    std::string device_;
    std::array<int16_t, 8> axes_{};
    std::array<uint8_t, 16> buttons_{};
    std::chrono::steady_clock::time_point last_open_attempt_{};
    unitree::common::UnitreeJoystick joystick_{};
};

} // namespace isaaclab
