import random
import sqlite3
from typing import Dict, List, Optional

from states.agent_state import AgentState
from .diacritizer import Diacritizer

from .utils import print_tool_output

# Mapping Arabic phrases based on the number of verses
verse_mapping = {
    1: "بيت واحد",
    2: "بيتين",
    3: "ثلاثة أبيات",
    4: "أربعة أبيات",
    5: "خمسة أبيات",
    6: "ستة أبيات",
    7: "سبعة أبيات",
    8: "ثمانية أبيات",
    9: "تسعة أبيات",
    10: "عشرة أبيات",
    11: "أحد عشر بيتًا",
    12: "اثنا عشر بيتًا",
    13: "ثلاثة عشر بيتًا",
    14: "أربعة عشر بيتًا",
    15: "خمسة عشر بيتًا",
    16: "ستة عشر بيتًا",
    17: "سبعة عشر بيتًا",
    18: "ثمانية عشر بيتًا",
    19: "تسعة عشر بيتًا",
    20: "عشرون بيتًا",
}


class PoemRetrieval:
    def __init__(self, db_path: str = "data/arabic_poems.db"):
        self.db_path = db_path
        self.MAX_REFERENCE_POEMS = 20

    def search_poems(self, state: AgentState) -> List[Dict]:
        """
        Search for poems with hierarchical filtering:
        1. Try exact match with all criteria
        2. Try exact match with poet and era, flexible verse count
        3. Try same era but different poet
        4. Try different era
        5. Try only meter and rhyme with exact verse count
        6. Try meter and rhyme with any even verse count between 4-20
        """
        preference = state["user_preferences"]
        meter = preference["meter"]
        rhyme = preference["rhyme"]
        num_verses = preference["num_verses"]
        theme = preference.get("theme")
        poet = preference.get("poet_name")
        era = preference.get("era")

        # Limit number of reference poems to 20
        limit = min(state["num_reference_poems"], self.MAX_REFERENCE_POEMS)

        poems = []

        # Define search strategies in order of preference
        search_strategies = [
            # Strategy 1: Exact match with all provided criteria, including meter
            {"poet": poet, "era": era, "theme": theme, "verse_strategy": "exact", "meter": meter}
            if poet and era and theme
            else None,
            # Strategy 3: Same era, any poet, with meter
            {"poet": None, "era": era, "theme": theme, "verse_strategy": "exact", "meter": meter} if era else None,
            # Strategy 4: Any era, with theme, with meter
            {"poet": None, "era": None, "theme": theme, "verse_strategy": "exact", "meter": meter} if theme else None,
            # Strategy 2: Same poet and era, flexible verse count, with meter
            {"poet": poet, "era": era, "theme": theme, "verse_strategy": "flexible", "meter": meter}
            if poet and era and theme
            else None,
            # Strategy 3: Same era, any poet, flexible verse count, with meter
            {"poet": None, "era": era, "theme": theme, "verse_strategy": "flexible", "meter": meter} if era else None,
            # Strategy 4: Any era, with theme, flexible verse count, with meter
            {"poet": None, "era": None, "theme": theme, "verse_strategy": "flexible", "meter": meter}
            if theme
            else None,
            # Strategy 5: Any theme, with era, flexible verse count, with meter
            {"poet": None, "era": era, "theme": None, "verse_strategy": "flexible", "meter": meter} if era else None,
            # Strategy 5: Only rhyme and smart verse matching, no meter as all other params are None
            {"poet": None, "era": None, "theme": None, "verse_strategy": "flexible", "meter": None},
        ]

        # Remove None strategies
        search_strategies = [s for s in search_strategies if s is not None]

        for strategy in search_strategies:
            poems = self._execute_search(
                meter=strategy.get("meter"),
                rhyme=rhyme,
                num_verses=num_verses,
                poet=strategy["poet"],
                era=strategy["era"],
                theme=strategy["theme"],
                verse_strategy=strategy["verse_strategy"],
                limit=limit,
            )

            if len(poems) >= limit:  # Only break if we have enough poems
                break

        return {"reference_poems": poems}

    def _execute_search(
        self,
        meter: str,
        rhyme: str,
        num_verses: int,
        poet: Optional[str],
        era: Optional[str],
        theme: Optional[str],
        verse_strategy: str,
        limit: int,
    ) -> List[Dict]:
        conditions = ["meter = ?", "rhyme = ?"]
        params = [meter, rhyme]

        # Adjust verse strategy
        if verse_strategy == "exact":
            conditions.append("num_verses = ?")
            params.append(num_verses)
        elif verse_strategy == "flexible":
            lower_bound = max(4, ((num_verses - 1) // 2) * 2)
            upper_bound = min(20, ((num_verses + 2) // 2) * 2)
            conditions.append("num_verses BETWEEN ? AND ? AND num_verses % 2 = 0")
            params.extend([lower_bound, upper_bound])

        # Add additional filters
        if poet:
            conditions.append("poet = ?")
            params.append(poet)
        if era:
            conditions.append("era = ?")
            params.append(era)
        if theme:
            conditions.append("theme = ?")
            params.append(theme)

        where_clause = " AND ".join(conditions)
        base_query = f"""
            SELECT p.*,
                   GROUP_CONCAT(v.first_hemistich, '|') AS first_hemistiches,
                   GROUP_CONCAT(v.second_hemistich, '|') AS second_hemistiches
            FROM poems p
            JOIN verses v ON p.id = v.poem_id
            WHERE {where_clause}
            GROUP BY p.id
        """

        # For flexible verse strategy, prioritize by proximity to desired num_verses
        if verse_strategy == "flexible":
            order_by = f"ABS(num_verses - {num_verses})"
            base_query += f" ORDER BY {order_by}"

        initial_fetch_limit = limit * 3
        query = f"{base_query} LIMIT ?"
        params.append(initial_fetch_limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        all_poems = [self._format_poem(dict(row)) for row in rows]

        # Apply random sampling if more poems than the limit
        if len(all_poems) > limit:
            selected_poems = random.sample(all_poems, limit)
        else:
            selected_poems = all_poems

        for p in selected_poems:
            print_tool_output(output=p["llm_prompt"], tool="PoemRetrieval")

        return selected_poems

    def _format_poem(self, row: Dict) -> Dict:
        """Format a poem for LLM prompt generation with diacritization"""
        diacritizer = Diacritizer(
            weights_path="/home/khaled/workspace/projects/allam/models/weights/diacritizer_model_weights.pt"
        )

        first_hemistiches = row["first_hemistiches"].split("|")
        second_hemistiches = row["second_hemistiches"].split("|")

        # Process and diacritize each verse
        formatted_verses = []
        for first, second in zip(first_hemistiches, second_hemistiches):
            # Create verse state for diacritizer
            verse_state = {"verse": {"first_hemistich": first.strip(), "second_hemistich": second.strip()}}

            # Get diacritized hemistiches
            diacritized = diacritizer(verse_state)
            diacritized_verse = diacritized["verse"]

            formatted_verse = f"{diacritized_verse['first_hemistich']} - {diacritized_verse['second_hemistich']}"
            formatted_verses.append(formatted_verse)

        formatted_poem = f"""Input:
الشاعر: {row['poet']}
العصر: {row['era']}
البحر: {row['meter']}
القافية: {row['rhyme_type']}
الموضوع: {row.get('theme', 'غير محدد')}
عدد الأبيات: {row['num_verses']}
Output:
[بداية_القصيدة]
{chr(10).join(formatted_verses)}
[نهاية_القصيدة]"""

        return {
            "metadata": {
                "meter": row["meter"],
                "theme": row["theme"],
                "era": row["era"],
                "poet": row["poet"],
                "rhyme": row["rhyme"],
                "rhyme_type": row["rhyme_type"],
                "num_verses": row["num_verses"],
            },
            "verses": formatted_verses,
            "llm_prompt": formatted_poem,
        }
