import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"research_log"))
from cycle009_analysis import choose_candidate,transitions


def candidate(cmsa,miou,clean_miou=.92,clean_cmsa=.96):
    return {"clean":{"target_miou":clean_miou,"cmsa":clean_cmsa},"target15_b":{"target_miou":miou,"cmsa":cmsa}}


def test_identity_priority_clean_preservation_and_small_tie():
    base={"num_groups":50,"target_miou":.92,"cmsa":.94}
    old=candidate(.62,.713)
    assert choose_candidate(base,{"cycle008":old,"no_cons":candidate(.64,.69)})[0]=="no_cons"
    assert choose_candidate(base,{"cycle008":old,"no_cons":candidate(.64,.75,clean_miou=.90)})[0]=="cycle008"
    assert choose_candidate(base,{"cycle008":old,"no_cons":candidate(.62,.714)})[0]=="cycle008"
    assert choose_candidate(base,{"cycle008":old,"no_cons":candidate(.62,.717)})[0]=="no_cons"


def test_transition_counts_preserve_direction():
    assert transitions([False,True,True,False],[True,False,True,False])=={"fail_to_pass":1,"pass_to_fail":1,"pass_to_pass":1,"fail_to_fail":1}
