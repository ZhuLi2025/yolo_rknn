#!/bin/bash
# 实时看 RK3588 三颗 NPU 核的占用。
# 每核百分比在 /sys/kernel/debug/rknpu/load，这个 debugfs 只有 root 能读。
# 直接运行即可；不是 root 时会 sudo 再拉起自己（会要一次密码）。
#
#   scripts/npu_load.sh          # 每 0.5 秒刷新
#   scripts/npu_load.sh 1        # 每 1 秒刷新
#   Ctrl-C 退出

set -u

interval="${1:-0.5}"
load_path="/sys/kernel/debug/rknpu/load"
devfreq="/sys/class/devfreq/fdab0000.npu"

if [[ ! -r "${load_path}" ]]; then
    if [[ "$(id -u)" -ne 0 ]]; then
        echo "每核负载在 ${load_path}，需要 root，改用 sudo 重新启动。"
        exec sudo -E "$0" "$@"
    fi
    echo "读不到 ${load_path}。debugfs 没挂上，或者这颗内核没有 RKNPU debug 节点。"
    exit 1
fi

npu_temp_path=""
for zone in /sys/class/thermal/thermal_zone*; do
    if [[ -r "${zone}/type" ]] && [[ "$(cat "${zone}/type")" == "npu-thermal" ]]; then
        npu_temp_path="${zone}/temp"
        break
    fi
done

bar() {
    local pct="$1"
    local width=20
    local filled=$((pct * width / 100))
    local i
    local out=""
    for ((i = 0; i < width; ++i)); do
        if ((i < filled)); then
            out+="#"
        else
            out+="."
        fi
    done
    printf "%s" "${out}"
}

cleanup() {
    printf "\n"
    exit 0
}
trap cleanup INT TERM

while true; do
    raw="$(cat "${load_path}" 2>/dev/null || true)"
    freq="-"
    agg="-"
    if [[ -r "${devfreq}/cur_freq" ]]; then
        hz="$(cat "${devfreq}/cur_freq")"
        freq="$((hz / 1000000)) MHz"
    fi
    if [[ -r "${devfreq}/load" ]]; then
        agg="$(cat "${devfreq}/load")"
    fi
    temp="-"
    if [[ -n "${npu_temp_path}" && -r "${npu_temp_path}" ]]; then
        milli="$(cat "${npu_temp_path}")"
        temp="$(awk -v m="${milli}" 'BEGIN { printf "%.1f C", m / 1000 }')"
    fi

    printf "\033[H\033[2J"
    printf "RK3588 NPU   %s   freq %s   temp %s   devfreq %s\n" "$(date '+%H:%M:%S')" "${freq}" "${temp}" "${agg}"
    printf "batch_size=2 时 core0 是贴起始边的那张，core1 是贴结束边的那张，core2 空闲。\n\n"

    found=0
    while IFS= read -r token; do
        core="${token%%:*}"
        pct="${token#*:}"
        pct="${pct//%/}"
        pct="${pct// /}"
        if [[ ! "${pct}" =~ ^[0-9]+$ ]]; then
            continue
        fi
        if ((pct > 100)); then
            pct=100
        fi
        printf "  %-6s %3d%%  %s\n" "${core}" "${pct}" "$(bar "${pct}")"
        found=1
    done < <(grep -oE 'Core[0-9]+:[[:space:]]*[0-9]+%' <<<"${raw}" || true)

    if [[ "${found}" -eq 0 ]]; then
        printf "未能解析每核百分比，原始内容：\n%s\n" "${raw}"
    fi
    printf "\n刷新间隔 %s 秒，Ctrl-C 退出\n" "${interval}"
    sleep "${interval}"
done
