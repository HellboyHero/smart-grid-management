from ..utils.supabase_client import get_supabase_client

class BillService:
    def __init__(self):
        self.supabase = get_supabase_client()

    def store_bill(self, user_id, bill_data):
        """Store bill information in Supabase"""
        try:
            response = self.supabase.table('bills').insert({
                'user_id': str(user_id),
                'bill_amount': float(bill_data['bill_amount']),
                'bill_period': bill_data['bill_period'],
                'bill_units': int(bill_data['bill_units']),
                'bills_generated': int(bill_data['bills_generated'])
            }).execute()
            
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error storing bill: {e}")
            return None

    def get_bills_count(self, user_id):
        """Get the total number of bills for a user"""
        try:
            response = self.supabase.table('bills')\
                .select('id', count='exact')\
                .eq('user_id', str(user_id))\
                .execute()
            
            return response.count if response.count is not None else 0
        except Exception as e:
            print(f"Error counting bills: {e}")
            return 0

    def get_all_bills(self, user_id):
        """Get all bills for a user"""
        try:
            response = self.supabase.table('bills')\
                .select('*')\
                .eq('user_id', str(user_id))\
                .order('created_at', desc=True)\
                .execute()
            
            return response.data
        except Exception as e:
            print(f"Error fetching bills: {e}")
            return [] 