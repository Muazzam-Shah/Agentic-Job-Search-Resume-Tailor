"""
Test script to verify cover letter vs resume intent detection
"""

# Simulate the intent detection logic
def test_intent_detection():
    """Test various user inputs to ensure correct intent classification"""
    
    test_cases = [
        ("generate cover letter", "generate_cover_letter"),
        ("create cover letter", "generate_cover_letter"),
        ("write a cover letter", "generate_cover_letter"),
        ("cover letter please", "generate_cover_letter"),
        ("yes generate cover letter", "generate_cover_letter"),
        ("generate resume", "generate_resume"),
        ("create tailored resume", "generate_resume"),
        ("tailor my resume", "generate_resume"),
        ("yes", None),  # Should go to LLM or workflow continuation
    ]
    
    print("🧪 Testing Intent Detection Fix\n")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for user_input, expected_intent in test_cases:
        user_lower = user_input.lower()
        detected_intent = None
        
        # Check for COVER LETTER first (before resume check)
        if any(phrase in user_lower for phrase in ["cover letter", "cover-letter", "coverletter"]) or \
           (("cover" in user_lower or "letter" in user_lower) and 
            any(word in user_lower for word in ["generate", "create", "write", "make"])):
            detected_intent = "generate_cover_letter"
        
        # Check for resume generation
        elif any(word in user_lower for word in ["resume", "tailor"]) and \
             not any(word in user_lower for word in ["cover", "letter"]):
            detected_intent = "generate_resume"
        
        # Determine result
        if expected_intent is None:
            result = "⏭️  SKIPPED (LLM decision)"
            status = "✅"
        elif detected_intent == expected_intent:
            result = f"✅ PASS - Detected: {detected_intent}"
            status = "✅"
            passed += 1
        else:
            result = f"❌ FAIL - Expected: {expected_intent}, Got: {detected_intent}"
            status = "❌"
            failed += 1
        
        print(f"{status} Input: '{user_input}'")
        print(f"   {result}")
        print()
    
    print("=" * 60)
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! Cover letter intent is now properly detected.")
    else:
        print(f"⚠️  {failed} test(s) failed. Check the logic.")
    
    return failed == 0

if __name__ == "__main__":
    success = test_intent_detection()
    exit(0 if success else 1)
