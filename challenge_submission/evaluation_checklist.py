"""Quick checklist for the CTR challenge deliverables.

Run this script to print the items you must cover in the report/presentation.
"""

CHECKLIST = {
    "Policy": [
        "외부 구현 사용 시 반드시 출처 명시 (미기재 시 20점 감점)",
    ],
    "EDA (10점)": [
        "데이터 분포, 결측치, 시퀀스 길이 등 탐색",
        "시각화/요약 표 준비",
    ],
    "Feature Selection (5점)": [
        "사용 피처와 신규 파생 피처 선정 근거 작성",
    ],
    "Model Selection (5점)": [
        "모델 구조·선택 이유 설명",
    ],
    "Novelty (20점)": [
        "모델 변경 2점",
        "피처 분석/신규 피처 3~6점",
        "앙상블 2점",
        "하이퍼파라미터 변경 3점",
        "WOW 요소 0~10점",
    ],
    "Ranking": [
        "최종 순위 및 점수 기록",
    ],
}


def print_checklist():
    print("== CTR Challenge Evaluation Checklist ==")
    for section, items in CHECKLIST.items():
        print(f"\n[{section}]")
        for bullet in items:
            print(f" - {bullet}")


if __name__ == "__main__":
    print_checklist()
