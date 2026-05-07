import unittest

from app.services.database.query_validator import QueryValidator


class QueryValidatorPrivacyTests(unittest.TestCase):
    def assert_rejected(self, sql):
        ok, error = QueryValidator.is_read_only(sql)
        self.assertFalse(ok, msg=sql)
        self.assertTrue(error)

    def assert_allowed(self, sql):
        ok, error = QueryValidator.is_read_only(sql)
        self.assertTrue(ok, msg=error)

    def test_rejects_direct_private_table_and_star(self):
        self.assert_rejected("SELECT * FROM users")
        self.assert_rejected("SHOW TABLES")
        self.assert_rejected("DESCRIBE users")

    def test_rejects_private_columns_even_when_aliased(self):
        self.assert_rejected("SELECT email AS seller FROM users WHERE deleted_at IS NULL")
        self.assert_rejected(
            "SELECT offline_buyer_mobile AS mobile "
            "FROM buy_sell_products WHERE deleted_at IS NULL"
        )

    def test_allows_public_product_query(self):
        self.assert_allowed(
            "SELECT name, price, discount_price "
            "FROM kshop_products "
            "WHERE deleted_at IS NULL AND status = 1 LIMIT 10"
        )

    def test_allows_join_only_user_name(self):
        self.assert_allowed(
            "SELECT bp.product_name, bp.price, u.name AS seller "
            "FROM buy_sell_products bp "
            "LEFT JOIN users u ON bp.seller_id = u.id "
            "WHERE bp.deleted_at IS NULL LIMIT 10"
        )

    def test_rejects_join_only_user_contact(self):
        self.assert_rejected(
            "SELECT bp.product_name, u.mobile_number AS seller_mobile "
            "FROM buy_sell_products bp "
            "LEFT JOIN users u ON bp.seller_id = u.id "
            "WHERE bp.deleted_at IS NULL LIMIT 10"
        )


if __name__ == "__main__":
    unittest.main()

