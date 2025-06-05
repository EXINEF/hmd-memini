ADD_TASK_SLOTS = [
    {"task_name": "buy groceries", "due_date": "tomorrow"},
    {"task_name": "call mom", "due_date": "this weekend"},
    {"task_name": "submit report", "due_date": "Friday"},
    {"task_name": "schedule dentist appointment", "due_date": "next week"},
    {"task_name": "pay bills", "due_date": "by end of month"},
    {"task_name": None, "due_date": "tomorrow"},  # Missing task_name
    {"task_name": "prepare presentation", "due_date": None},
    {"task_name": "buy groceries", "due_date": "tomorrow"},  # Duplicate
]

COMPLETE_TASK_SLOTS = [
    {"task_name": "buy groceries"},
    {"task_name": "call mom"},
    {"task_name": "non-existent task"},  # Non-existent
    {"task_name": None},  # Missing task_name
    {"task_name": "buy"},  # Ambiguous, multiple matches
]

DELETE_TASK_SLOTS = [
    {"task_name": "buy groceries", "confirmation": True},
    {"task_name": "call mom", "confirmation": False},
    {"task_name": "non-existent task", "confirmation": None},  # Non-existent
    {"task_name": None, "confirmation": None},  # Missing task_name
    {"task_name": "buy", "confirmation": None},  # Ambiguous, multiple matches
]

RETRIEVE_TASKS_SLOTS = [{"status": "pending"}, {"status": "completed"}]

UPDATE_TASK_SLOTS = [
    {
        "task_name": "buy groceries",
        "new_task_name": "buy food",
        "new_due_date": None,
    },
    {"task_name": "call mom", "new_task_name": None, "new_due_date": "Saturday"},
    {
        "task_name": "call mom",
        "new_task_name": "call parents",
        "new_due_date": "Sunday",
    },
    {
        "task_name": "non-existent task",
        "new_task_name": "new name",
        "new_due_date": "tomorrow",
    },
    {"task_name": None, "new_task_name": "new name", "new_due_date": "tomorrow"},
    {"task_name": "buy", "new_task_name": "new name", "new_due_date": "tomorrow"},
    {
        "task_name": "prepare presentation",
        "new_task_name": None,
        "new_due_date": None,
    },
]


def analyze_nlu_test_data_statistics(test_data):
    """Generate comprehensive statistics about the test data."""
    stats = {
        "dataset_overview": {},
        "slot_distribution": {},
        "utterance_characteristics": {},
        "template_usage": {},
        "vocabulary_analysis": {},
    }

    # Dataset Overview
    total_samples = sum(len(intent_data) for intent_data in test_data.values())
    stats["dataset_overview"] = {
        "total_samples": total_samples,
        "intents_count": len(test_data),
        "samples_per_intent": {intent: len(data) for intent, data in test_data.items()},
        "single_intent_samples": sum(
            len(data) for intent, data in test_data.items() if intent != "multi_intent"
        ),
        "multi_intent_samples": len(test_data.get("multi_intent", [])),
    }

    # Slot Distribution Analysis
    slot_counts = {}
    slot_value_diversity = {}

    for intent, intent_data in test_data.items():
        if intent == "multi_intent":
            # Handle multi-intent slots differently
            for sample in intent_data:
                if "slots" in sample:
                    for slot_group in sample["slots"]:
                        for slot, value in slot_group.items():
                            if slot not in slot_counts:
                                slot_counts[slot] = 0
                                slot_value_diversity[slot] = set()
                            slot_counts[slot] += 1
                            slot_value_diversity[slot].add(str(value))
        else:
            for sample in intent_data:
                if "slots" in sample:
                    for slot, value in sample["slots"].items():
                        if slot not in slot_counts:
                            slot_counts[slot] = 0
                            slot_value_diversity[slot] = set()
                        slot_counts[slot] += 1
                        slot_value_diversity[slot].add(str(value))

    stats["slot_distribution"] = {
        "slot_frequencies": slot_counts,
        "unique_values_per_slot": {
            slot: len(values) for slot, values in slot_value_diversity.items()
        },
        "slot_value_diversity": {
            slot: list(values) for slot, values in slot_value_diversity.items()
        },
    }

    # Utterance Characteristics
    all_utterances = []
    utterance_lengths = []

    for intent, intent_data in test_data.items():
        for sample in intent_data:
            utterance = sample.get("question", "")
            all_utterances.append(utterance)
            utterance_lengths.append(len(utterance.split()))

    stats["utterance_characteristics"] = {
        "total_utterances": len(all_utterances),
        "avg_length_words": round(sum(utterance_lengths) / len(utterance_lengths), 2),
        "min_length_words": min(utterance_lengths),
        "max_length_words": max(utterance_lengths),
        "length_distribution": {
            "short_utterances_1_5_words": len(
                [l for l in utterance_lengths if 1 <= l <= 5]
            ),
            "medium_utterances_6_10_words": len(
                [l for l in utterance_lengths if 6 <= l <= 10]
            ),
            "long_utterances_11plus_words": len(
                [l for l in utterance_lengths if l >= 11]
            ),
        },
    }

    template_coverage = {
        "add_task_templates": 10,  # From ADD_TASK_TEMPLATES
        "complete_task_templates": 10,  # From COMPLETE_TASK_TEMPLATES
        "delete_task_templates": 10,  # From DELETE_TASK_TEMPLATES
        "retrieve_tasks_templates": 10,  # From RETRIEVE_TASKS_TEMPLATES
        "update_task_templates": 10,  # From UPDATE_TASK_TEMPLATES
        "fallback_templates": 10,  # From FALLBACK_TEMPLATES
        "multi_intent_templates": 15,  # From MULTI_INTENT_TEMPLATES
    }

    stats["template_usage"] = {
        "total_templates_available": sum(template_coverage.values()),
        "templates_per_intent": template_coverage,
        "generation_method": "random_selection_with_replacement",
    }

    # Vocabulary Analysis
    all_words = []
    for utterance in all_utterances:
        all_words.extend(utterance.lower().split())

    unique_words = set(all_words)
    word_freq = {}
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    stats["vocabulary_analysis"] = {
        "total_words": len(all_words),
        "unique_words": len(unique_words),
        "vocabulary_diversity_ratio": round(len(unique_words) / len(all_words), 3),
        "most_common_words": sorted(
            word_freq.items(), key=lambda x: x[1], reverse=True
        )[:10],
    }

    return stats


def print_nlu_data_statistics(stats):
    """Print formatted statistics report."""
    print("\n" + "=" * 50)
    print("DATASET STATISTICS REPORT")
    print("=" * 50)

    print(f"\n📊 DATASET OVERVIEW:")
    overview = stats["dataset_overview"]
    print(f"  • Total samples: {overview['total_samples']}")
    print(f"  • Number of intents: {overview['intents_count']}")
    print(f"  • Single-intent samples: {overview['single_intent_samples']}")
    print(f"  • Multi-intent samples: {overview['multi_intent_samples']}")
    print(f"  • Samples per intent:")
    for intent, count in overview["samples_per_intent"].items():
        print(f"    - {intent}: {count}")

    print(f"\n🎯 SLOT DISTRIBUTION:")
    slot_dist = stats["slot_distribution"]
    print(f"  • Slot frequencies:")
    for slot, freq in slot_dist["slot_frequencies"].items():
        unique_vals = slot_dist["unique_values_per_slot"][slot]
        print(f"    - {slot}: {freq} occurrences, {unique_vals} unique values")

    print(f"\n💬 UTTERANCE CHARACTERISTICS:")
    utterance = stats["utterance_characteristics"]
    print(f"  • Total utterances: {utterance['total_utterances']}")
    print(f"  • Average length: {utterance['avg_length_words']} words")
    print(
        f"  • Length range: {utterance['min_length_words']}-{utterance['max_length_words']} words"
    )
    print(f"  • Length distribution:")
    for category, count in utterance["length_distribution"].items():
        print(f"    - {category.replace('_', ' ').title()}: {count}")

    print(f"\n📝 TEMPLATE COVERAGE:")
    template = stats["template_usage"]
    print(f"  • Total templates available: {template['total_templates_available']}")
    print(f"  • Templates per intent:")
    for intent, count in template["templates_per_intent"].items():
        print(f"    - {intent}: {count} templates")

    print(f"\n📚 VOCABULARY ANALYSIS:")
    vocab = stats["vocabulary_analysis"]
    print(f"  • Total words: {vocab['total_words']}")
    print(f"  • Unique words: {vocab['unique_words']}")
    print(f"  • Vocabulary diversity: {vocab['vocabulary_diversity_ratio']}")
    print(
        f"  • Most common words: {', '.join([word for word, freq in vocab['most_common_words'][:5]])}"
    )


def analyze_dm_test_data_statistics():
    # Test database tasks
    DB_TASKS = [
        {"task_name": "buy groceries", "due_date": "tomorrow", "completed": False},
        {"task_name": "call mom", "due_date": "this weekend", "completed": False},
        {"task_name": "submit report", "due_date": "Friday", "completed": False},
        {"task_name": "buy milk", "due_date": "today", "completed": False},
        {"task_name": "buy eggs", "due_date": "today", "completed": True},
    ]

    # Organize test data
    test_data = {
        "add_task": ADD_TASK_SLOTS,
        "complete_task": COMPLETE_TASK_SLOTS,
        "delete_task": DELETE_TASK_SLOTS,
        "retrieve_tasks": RETRIEVE_TASKS_SLOTS,
        "update_task": UPDATE_TASK_SLOTS,
    }

    stats = {
        "dataset_overview": {},
        "slot_distribution": {},
        "test_scenario_analysis": {},
        "action_distribution": {},
        "database_setup": {},
        "test_complexity": {},
    }

    # Dataset Overview
    total_samples = sum(len(intent_data) for intent_data in test_data.values())
    stats["dataset_overview"] = {
        "total_samples": total_samples,
        "intents_count": len(test_data),
        "samples_per_intent": {intent: len(data) for intent, data in test_data.items()},
        "database_tasks": len(DB_TASKS),
    }

    # Slot Distribution Analysis
    slot_counts = {}
    slot_value_diversity = {}

    for intent, intent_data in test_data.items():
        for sample in intent_data:
            for slot, value in sample.items():
                if slot not in slot_counts:
                    slot_counts[slot] = 0
                    slot_value_diversity[slot] = set()
                slot_counts[slot] += 1
                if value is not None:
                    slot_value_diversity[slot].add(str(value))
                else:
                    slot_value_diversity[slot].add("None")

    stats["slot_distribution"] = {
        "slot_frequencies": slot_counts,
        "unique_values_per_slot": {
            slot: len(values) for slot, values in slot_value_diversity.items()
        },
        "slot_value_diversity": {
            slot: list(values) for slot, values in slot_value_diversity.items()
        },
    }

    # Test Scenario Analysis
    scenario_counts = {
        "normal_cases": 0,
        "missing_required_slots": 0,
        "non_existent_references": 0,
        "ambiguous_references": 0,
        "duplicate_cases": 0,
        "confirmation_cases": 0,
    }

    for intent, intent_data in test_data.items():
        for sample in intent_data:
            # Check for missing required slots
            if None in sample.values():
                scenario_counts["missing_required_slots"] += 1
            # Check for non-existent references
            elif any("non-existent" in str(v) for v in sample.values() if v):
                scenario_counts["non_existent_references"] += 1
            # Check for ambiguous references (like "buy" matching multiple tasks)
            elif any(str(v) == "buy" for v in sample.values() if v):
                scenario_counts["ambiguous_references"] += 1
            # Check for duplicates
            elif intent == "add_task" and sample.get("task_name") == "buy groceries":
                scenario_counts["duplicate_cases"] += 1
            # Check for confirmation scenarios
            elif "confirmation" in sample:
                scenario_counts["confirmation_cases"] += 1
            else:
                scenario_counts["normal_cases"] += 1

    stats["test_scenario_analysis"] = scenario_counts

    # Expected Action Distribution
    action_types = {
        "add_task": 0,
        "complete_task": 0,
        "delete_task": 0,
        "update_task": 0,
        "retrieve_tasks": 0,
        "req_task_name": 0,
        "req_task_update_details": 0,
        "confirm_delete_task": 0,
        "task_not_found": 0,
        "task_already_exists": 0,
        "multiple_tasks_found": 0,
    }

    # Count expected actions based on test cases
    for intent, intent_data in test_data.items():
        for sample in intent_data:
            expected_actions = determine_expected_actions_for_stats(intent, sample)
            for action in expected_actions:
                if action.lower().replace("_", "_") in action_types:
                    action_types[action.lower().replace("_", "_")] += 1

    stats["action_distribution"] = action_types

    # Database Setup Analysis
    db_stats = {
        "total_tasks": len(DB_TASKS),
        "pending_tasks": len([t for t in DB_TASKS if not t["completed"]]),
        "completed_tasks": len([t for t in DB_TASKS if t["completed"]]),
        "unique_task_names": len(set(t["task_name"] for t in DB_TASKS)),
        "tasks_with_same_prefix": len(
            [t for t in DB_TASKS if t["task_name"].startswith("buy")]
        ),
        "due_date_variety": len(set(t["due_date"] for t in DB_TASKS)),
    }

    stats["database_setup"] = db_stats

    # Test Complexity Analysis
    complexity_analysis = {
        "simple_tests": 0,  # Direct slot matching
        "edge_case_tests": 0,  # Missing slots, non-existent, etc.
        "ambiguity_tests": 0,  # Multiple matches, unclear references
        "state_dependent_tests": 0,  # Tests that depend on DB state
    }

    for intent, intent_data in test_data.items():
        for sample in intent_data:
            if None in sample.values():
                complexity_analysis["edge_case_tests"] += 1
            elif any("non-existent" in str(v) for v in sample.values() if v):
                complexity_analysis["edge_case_tests"] += 1
            elif any(str(v) == "buy" for v in sample.values() if v):
                complexity_analysis["ambiguity_tests"] += 1
            elif intent == "add_task" and sample.get("task_name") in [
                t["task_name"] for t in DB_TASKS
            ]:
                complexity_analysis["state_dependent_tests"] += 1
            else:
                complexity_analysis["simple_tests"] += 1

    stats["test_complexity"] = complexity_analysis

    return stats


def determine_expected_actions_for_stats(intent: str, slots: dict) -> list:
    """Helper function to determine expected actions for statistics."""
    actions = []

    if intent == "add_task":
        if slots.get("task_name") is None:
            actions.append("req_task_name")
        elif slots.get("task_name") == "buy groceries":
            actions.append("task_already_exists")
        else:
            actions.append("add_task")
    elif intent == "complete_task":
        if slots.get("task_name") is None:
            actions.append("req_task_name")
        elif slots.get("task_name") == "non-existent task":
            actions.append("task_not_found")
        elif slots.get("task_name") == "buy":
            actions.append("multiple_tasks_found")
        else:
            actions.append("complete_task")
    elif intent == "delete_task":
        if slots.get("task_name") is None:
            actions.append("req_task_name")
        elif slots.get("task_name") == "non-existent task":
            actions.append("task_not_found")
        elif slots.get("task_name") == "buy":
            actions.append("multiple_tasks_found")
        elif slots.get("confirmation") is True:
            actions.append("delete_task")
        else:
            actions.append("confirm_delete_task")
    elif intent == "retrieve_tasks":
        actions.append("retrieve_tasks")
    elif intent == "update_task":
        if slots.get("task_name") is None:
            actions.append("req_task_name")
        elif slots.get("task_name") == "non-existent task":
            actions.append("task_not_found")
        elif slots.get("task_name") == "buy":
            actions.append("multiple_tasks_found")
        elif slots.get("new_task_name") is None and slots.get("new_due_date") is None:
            actions.append("req_task_update_details")
        else:
            actions.append("update_task")

    return actions


def print_dm_data_statistics(stats):
    """Print formatted DM statistics report."""
    print("\n" + "=" * 60)
    print("DIALOGUE MANAGER DATASET STATISTICS REPORT")
    print("=" * 60)

    # Dataset Overview
    print(f"\n📊 DATASET OVERVIEW:")
    overview = stats["dataset_overview"]
    print(f"  • Total test samples: {overview['total_samples']}")
    print(f"  • Number of intents: {overview['intents_count']}")
    print(f"  • Database tasks for testing: {overview['database_tasks']}")
    print(f"  • Samples per intent:")
    for intent, count in overview["samples_per_intent"].items():
        print(f"    - {intent}: {count}")

    # Slot Distribution
    print(f"\n🎯 SLOT DISTRIBUTION:")
    slot_dist = stats["slot_distribution"]
    print(f"  • Slot frequencies:")
    for slot, freq in slot_dist["slot_frequencies"].items():
        unique_vals = slot_dist["unique_values_per_slot"][slot]
        print(f"    - {slot}: {freq} occurrences, {unique_vals} unique values")

    # Test Scenario Analysis
    print(f"\n🧪 TEST SCENARIO ANALYSIS:")
    scenarios = stats["test_scenario_analysis"]
    total_scenarios = sum(scenarios.values())
    print(f"  • Total test scenarios: {total_scenarios}")
    for scenario, count in scenarios.items():
        percentage = (count / total_scenarios * 100) if total_scenarios > 0 else 0
        print(
            f"    - {scenario.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
        )

    # Expected Action Distribution
    print(f"\n⚡ EXPECTED ACTION DISTRIBUTION:")
    actions = stats["action_distribution"]
    total_actions = sum(actions.values())
    print(f"  • Total expected actions: {total_actions}")
    for action, count in actions.items():
        if count > 0:
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            print(
                f"    - {action.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
            )

    # Database Setup
    print(f"\n🗄️ TEST DATABASE SETUP:")
    db_stats = stats["database_setup"]
    print(f"  • Total tasks in test DB: {db_stats['total_tasks']}")
    print(f"  • Pending tasks: {db_stats['pending_tasks']}")
    print(f"  • Completed tasks: {db_stats['completed_tasks']}")
    print(f"  • Unique task names: {db_stats['unique_task_names']}")
    print(
        f"  • Tasks with 'buy' prefix: {db_stats['tasks_with_same_prefix']} (for ambiguity testing)"
    )
    print(f"  • Due date variety: {db_stats['due_date_variety']} different due dates")

    # Test Complexity
    print(f"\n🎯 TEST COMPLEXITY ANALYSIS:")
    complexity = stats["test_complexity"]
    total_complexity = sum(complexity.values())
    print(f"  • Test complexity distribution:")
    for test_type, count in complexity.items():
        percentage = (count / total_complexity * 100) if total_complexity > 0 else 0
        print(
            f"    - {test_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
        )
