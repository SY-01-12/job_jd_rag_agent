from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


class GapAnalysisInput(BaseModel):
    jd_skills: List[str] = Field(..., description="岗位要求技能列表")
    user_skills: List[str] = Field(..., description="用户已掌握技能列表")


def analyze_skill_gap_func(jd_skills: List[str], user_skills: List[str]) -> dict:
    """
    分析岗位技能与用户技能之间的差距。
    """
    jd_set = {skill.strip().lower() for skill in jd_skills if skill.strip()}
    user_set = {skill.strip().lower() for skill in user_skills if skill.strip()}

    matched = sorted(jd_set & user_set)
    missing = sorted(jd_set - user_set)

    match_ratio = round(len(matched) / len(jd_set), 4) if jd_set else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_ratio": match_ratio,
        "summary": f"岗位要求 {len(jd_set)} 项技能，已匹配 {len(matched)} 项，缺失 {len(missing)} 项。"
    }


gap_analysis_tool = StructuredTool.from_function(
    func=analyze_skill_gap_func,
    name="analyze_skill_gap",
    description="对比岗位技能和用户技能，输出已匹配技能、缺失技能和匹配比例。",
    args_schema=GapAnalysisInput,
)