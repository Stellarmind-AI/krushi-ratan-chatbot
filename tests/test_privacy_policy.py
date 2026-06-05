import unittest

from app.utils.privacy_policy import get_privacy_policy


class PrivacyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = get_privacy_policy()

    def test_sensitive_current_tables_are_excluded(self):
        for table in [
            "users",
            "user_otps",
            "notifications",
            "settings",
            "farmer_orders",
            "company_orders",
            "kshop_orders",
        ]:
            if table == "users":
                self.assertTrue(self.policy.is_join_only_table(table))
                self.assertFalse(self.policy.is_queryable_table(table))
            else:
                self.assertFalse(self.policy.is_queryable_table(table))

    def test_future_operational_tables_are_denied(self):
        for table in [
            "sessions",
            "password_reset_tokens",
            "jobs",
            "content_reports",
            "schema_migrations",
        ]:
            self.assertFalse(self.policy.is_queryable_table(table))

    def test_users_join_only_allows_name_only(self):
        self.assertTrue(self.policy.is_join_only_table("users"))
        self.assertTrue(self.policy.is_safe_selected_column("users", "name"))
        self.assertFalse(self.policy.is_safe_selected_column("users", "email"))
        self.assertFalse(self.policy.is_safe_selected_column("users", "mobile_number"))

    def test_public_tool_columns_remove_private_fields(self):
        self.assertFalse(self.policy.is_safe_tool_column("buy_sell_products", "seller_id"))
        self.assertFalse(self.policy.is_safe_tool_column("buy_sell_products", "offline_buyer_mobile"))
        self.assertTrue(self.policy.is_safe_tool_column("buy_sell_products", "product_name"))


if __name__ == "__main__":
    unittest.main()

