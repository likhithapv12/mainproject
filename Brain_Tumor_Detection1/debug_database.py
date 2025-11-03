#!/usr/bin/env python3
"""
Database debugging script for Brain Tumor Detection System
Run this to check and fix database issues
"""

import sqlite3
import os
import json
from datetime import datetime

def check_database():
    """Check if database exists and has proper structure"""
    db_file = 'brain_tumor_detection.db'
    
    print("🔍 Checking database...")
    
    if not os.path.exists(db_file):
        print(f"❌ Database file '{db_file}' does not exist!")
        return False
    else:
        print(f"✅ Database file exists: {db_file}")
        print(f"   Size: {os.path.getsize(db_file)} bytes")
    
    return True

def check_tables():
    """Check database tables structure"""
    print("\n📋 Checking database tables...")
    
    try:
        conn = sqlite3.connect('brain_tumor_detection.db')
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  ✅ {table[0]}")
        
        # Check users table
        if ('users',) in tables:
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            print(f"  📊 Users table: {user_count} records")
            
            # Show sample user
            cursor.execute("SELECT id, username, full_name FROM users LIMIT 1")
            sample_user = cursor.fetchone()
            if sample_user:
                print(f"  👤 Sample user: ID={sample_user[0]}, Username={sample_user[1]}, Name={sample_user[2]}")
        
        # Check analysis_results table
        if ('analysis_results',) in tables:
            cursor.execute("SELECT COUNT(*) FROM analysis_results")
            result_count = cursor.fetchone()[0]
            print(f"  📊 Analysis results table: {result_count} records")
            
            if result_count > 0:
                cursor.execute("""SELECT id, user_id, patient_id, prediction, confidence, created_at 
                                  FROM analysis_results ORDER BY created_at DESC LIMIT 3""")
                recent_results = cursor.fetchall()
                print("  📋 Recent results:")
                for result in recent_results:
                    print(f"    - ID={result[0]}, User={result[1]}, Patient={result[2]}, Prediction={result[3]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
        return False

def recreate_database():
    """Recreate database with proper structure"""
    print("\n🔄 Recreating database...")
    
    try:
        # Remove existing database
        if os.path.exists('brain_tumor_detection.db'):
            os.remove('brain_tumor_detection.db')
            print("  🗑️ Removed old database")
        
        # Create new database
        conn = sqlite3.connect('brain_tumor_detection.db')
        c = conn.cursor()
        
        # Create users table
        c.execute('''CREATE TABLE users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      full_name TEXT NOT NULL,
                      username TEXT UNIQUE NOT NULL,
                      email TEXT UNIQUE NOT NULL,
                      password_hash TEXT NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        print("  ✅ Created users table")
        
        # Create analysis_results table
        c.execute('''CREATE TABLE analysis_results
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER NOT NULL,
                      patient_id TEXT NOT NULL,
                      image_name TEXT NOT NULL,
                      image_path TEXT NOT NULL,
                      prediction TEXT NOT NULL,
                      confidence REAL NOT NULL,
                      features TEXT NOT NULL,
                      is_tumor BOOLEAN NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        print("  ✅ Created analysis_results table")
        
        # Create sample user
        from werkzeug.security import generate_password_hash
        password_hash = generate_password_hash('medical123')
        
        c.execute('''INSERT INTO users (full_name, username, email, password_hash) 
                     VALUES (?, ?, ?, ?)''', 
                 ('Dr. John Smith', 'drjohn', 'john@hospital.com', password_hash))
        print("  ✅ Created sample user: drjohn / medical123")
        
        conn.commit()
        conn.close()
        
        print("✅ Database recreated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error recreating database: {e}")
        return False

def add_test_data():
    """Add test analysis results"""
    print("\n📊 Adding test data...")
    
    try:
        conn = sqlite3.connect('brain_tumor_detection.db')
        c = conn.cursor()
        
        # Get user ID
        c.execute("SELECT id FROM users WHERE username = 'drjohn'")
        user = c.fetchone()
        if not user:
            print("❌ Sample user not found")
            return False
        
        user_id = user[0]
        
        # Test data
        test_results = [
            {
                'patient_id': 'DRJOHN_TEST001',
                'image_name': 'sample_retinal_1.jpg',
                'image_path': 'static/uploads/sample_retinal_1.jpg',
                'prediction': 'Normal - No Tumor Indicators',
                'confidence': 92.5,
                'features': ['Normal optic disc appearance', 'Healthy RNFL', 'No papilledema'],
                'is_tumor': False
            },
            {
                'patient_id': 'DRJOHN_TEST002', 
                'image_name': 'sample_retinal_2.jpg',
                'image_path': 'static/uploads/sample_retinal_2.jpg',
                'prediction': 'Brain Tumor Indicators Detected',
                'confidence': 87.3,
                'features': ['Papilledema detected', 'RNFL thinning', 'Optic disc swelling'],
                'is_tumor': True
            }
        ]
        
        for result in test_results:
            c.execute('''INSERT INTO analysis_results 
                         (user_id, patient_id, image_name, image_path, prediction, confidence, features, is_tumor)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (user_id, result['patient_id'], result['image_name'], result['image_path'],
                      result['prediction'], result['confidence'], json.dumps(result['features']),
                      1 if result['is_tumor'] else 0))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Added {len(test_results)} test records")
        return True
        
    except Exception as e:
        print(f"❌ Error adding test data: {e}")
        return False

def verify_user_session():
    """Verify user session and database connection"""
    print("\n🔐 Checking user session and database connection...")
    
    try:
        conn = sqlite3.connect('brain_tumor_detection.db')
        conn.row_factory = sqlite3.Row
        
        # Get all users
        users = conn.execute('SELECT id, username, full_name FROM users').fetchall()
        print(f"All users in database:")
        for user in users:
            print(f"  ID={user['id']}, Username={user['username']}, Name={user['full_name']}")
        
        # Check analysis_results with user mapping
        results_with_users = conn.execute('''
            SELECT ar.id, ar.user_id, ar.patient_id, ar.prediction, ar.created_at, u.username
            FROM analysis_results ar
            LEFT JOIN users u ON ar.user_id = u.id
            ORDER BY ar.created_at DESC
        ''').fetchall()
        
        print(f"\nAll analysis results with user mapping:")
        if results_with_users:
            for result in results_with_users:
                print(f"  Result ID={result['id']}, User ID={result['user_id']}, Username={result['username'] or 'NULL'}, Patient={result['patient_id']}")
        else:
            print("  No analysis results found in database")
        
        # Test specific user query (simulate session)
        test_user = conn.execute('SELECT * FROM users WHERE username = ?', ('drjohn',)).fetchone()
        if test_user:
            user_results = conn.execute('''
                SELECT COUNT(*) as count FROM analysis_results WHERE user_id = ?
            ''', (test_user['id'],)).fetchone()
            print(f"\nTest query for user 'drjohn' (ID={test_user['id']}): {user_results['count']} results")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Session verification error: {e}")
        return False

def fix_database_associations():
    """Fix any broken user-result associations"""
    print("\n🔧 Fixing database associations...")
    
    try:
        conn = sqlite3.connect('brain_tumor_detection.db')
        
        # Check for orphaned results (results without valid user_id)
        orphaned = conn.execute('''
            SELECT ar.id, ar.user_id, ar.patient_id
            FROM analysis_results ar
            LEFT JOIN users u ON ar.user_id = u.id
            WHERE u.id IS NULL
        ''').fetchall()
        
        if orphaned:
            print(f"Found {len(orphaned)} orphaned results:")
            for result in orphaned:
                print(f"  Result ID={result[0]}, Invalid User ID={result[1]}, Patient={result[2]}")
            
            # Get the first valid user to reassign orphaned records
            valid_user = conn.execute('SELECT id FROM users LIMIT 1').fetchone()
            if valid_user:
                print(f"Reassigning orphaned records to user ID {valid_user[0]}...")
                conn.execute('UPDATE analysis_results SET user_id = ? WHERE user_id NOT IN (SELECT id FROM users)', (valid_user[0],))
                conn.commit()
                print("✅ Reassigned orphaned records")
        else:
            print("✅ No orphaned records found")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Fix associations error: {e}")
        return False

def main():
    """Main debugging function"""
    print("🔧 DATABASE DEBUG & REPAIR TOOL")
    print("=" * 50)
    
    # Check if database exists
    db_exists = check_database()
    
    if db_exists:
        # Check table structure
        tables_ok = check_tables()
        
        if tables_ok:
            # Verify user sessions and associations
            verify_user_session()
            
            # Fix any broken associations
            fix_database_associations()
            
            # Test operations
            operations_ok = test_database_operations()
            
            if operations_ok:
                print("\n🎉 Database is working correctly!")
                print("\n💡 If results still don't show under username:")
                print("1. Make sure you're logged in as the user who created the results")
                print("2. Check Flask console output when saving results")
                print("3. Verify session['user_id'] matches the database user_id")
                print("4. Try creating a new analysis result after running this script")
                return
    
    # If we get here, there are issues
    print("\n⚠️ Database issues detected. Attempting repair...")
    
    if recreate_database():
        add_test_data()
        
        print("\n✅ Database repair completed!")
        print("\n📋 Test the following:")
        print("1. Restart Flask app: python app.py")
        print("2. Login as: drjohn / medical123")
        print("3. Check Database tab - should show 2 test records")
        print("4. Upload new image and analyze")
        print("5. Save result and verify it appears in database")
        print("\n🎯 The username association should work now!")
    else:
        print("\n❌ Database repair failed!")
        print("Please check file permissions and try again.")

def test_database_operations():
    """Test basic database operations"""
    print("\n🧪 Testing database operations...")
    
    try:
        conn = sqlite3.connect('brain_tumor_detection.db')
        conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
        
        # Test user query
        user = conn.execute('SELECT * FROM users WHERE username = ?', ('drjohn',)).fetchone()
        if user:
            print(f"  ✅ User query works: {user['full_name']} (ID: {user['id']})")
        else:
            print("  ❌ User query failed")
            return False
        
        # Test analysis results query with JOIN
        results = conn.execute('''
            SELECT ar.patient_id, ar.prediction, ar.confidence, ar.features, ar.is_tumor,
                   u.username, u.full_name
            FROM analysis_results ar
            JOIN users u ON ar.user_id = u.id
            WHERE ar.user_id = ?
        ''', (user['id'],)).fetchall()
        
        print(f"  ✅ Results query with JOIN works: {len(results)} records found")
        
        for result in results:
            try:
                features = json.loads(result['features'])
                print(f"    - {result['patient_id']}: {result['prediction']} ({result['confidence']}%) - User: {result['username']}")
            except:
                print(f"    - {result['patient_id']}: {result['prediction']} (JSON parse error)")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Database operation error: {e}")
        return False

        if operations_ok:
                print("\n🎉 Database is working correctly!")
                print("\n💡 If save is still not working:")
                print("1. Check Flask console for error messages")
                print("2. Ensure you're logged in before saving")
                print("3. Check browser console (F12) for JavaScript errors")
                return
    
    # If we get here, there are issues
    print("\n⚠️ Database issues detected. Attempting repair...")
    
    if recreate_database():
        add_test_data()
        
        print("\n✅ Database repair completed!")
        print("\n📋 Next steps:")
        print("1. Restart your Flask app: python app.py")
        print("2. Login with: drjohn / medical123") 
        print("3. Try uploading and analyzing an image")
        print("4. The save should work now!")
    else:
        print("\n❌ Database repair failed!")
        print("Please check file permissions and try again.")

if __name__ == "__main__":
    main()