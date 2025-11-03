from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime
import uuid
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import random
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database initialization
def init_db():
    conn = sqlite3.connect('brain_tumor_detection.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  full_name TEXT NOT NULL,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Analysis results table
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_results
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
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect('brain_tumor_detection.db')
    conn.row_factory = sqlite3.Row
    return conn

# Simulate CNN model prediction
def simulate_cnn_prediction(image_path):
    """Simulate CNN model prediction with consistent results based on image content"""
    try:
        # Calculate a hash of the image content to ensure consistent results
        import hashlib
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_hash = hashlib.md5(image_data).hexdigest()
        
        # Use hash to create a deterministic "random" seed
        # This ensures the same image always gives the same result
        hash_int = int(image_hash[:8], 16)  # Use first 8 chars of hash as integer
        
        # Set random seed based on image hash
        import random
        random.seed(hash_int)
        
        # Now the "random" result will be consistent for the same image
        # Adjust probability based on simple image analysis
        with Image.open(image_path) as img:
            # Convert to grayscale for analysis
            gray_img = img.convert('L')
            pixels = np.array(gray_img)
            
            # Simple heuristic: darker images or images with high contrast 
            # are more likely to be flagged as abnormal
            mean_brightness = np.mean(pixels)
            contrast = np.std(pixels)
            
            # Base probability on image characteristics
            base_prob = 0.15  # 15% base chance
            
            # Adjust based on brightness (darker = higher chance)
            if mean_brightness < 100:
                base_prob += 0.15
            elif mean_brightness > 180:
                base_prob -= 0.05
                
            # Adjust based on contrast (higher contrast = higher chance)
            if contrast > 60:
                base_prob += 0.10
            elif contrast < 30:
                base_prob -= 0.05
                
            # Ensure probability stays within reasonable bounds
            base_prob = max(0.05, min(0.35, base_prob))
            
            # Make prediction based on adjusted probability
            is_tumor = random.random() < base_prob
        
        # Generate consistent confidence score (85-96%)
        confidence_seed = (hash_int % 1000) / 1000  # Normalize to 0-1
        confidence = round(85.0 + (confidence_seed * 11.0), 1)  # 85-96% range
        
        # Select features based on prediction and image characteristics
        if is_tumor:
            all_features = [
                'Papilledema detected - optic disc swelling present',
                'Increased cup-to-disc ratio observed',
                'Retinal nerve fiber layer (RNFL) thinning',
                'Abnormal vascular tortuosity patterns',
                'Signs of increased intracranial pressure',
                'Optic disc margin blurring detected',
                'Hemorrhages near optic disc',
                'Cotton wool spots identified'
            ]
        else:
            all_features = [
                'Normal optic disc appearance and margins',
                'Healthy retinal nerve fiber layer (RNFL)',
                'No signs of papilledema detected',
                'Normal cup-to-disc ratio (0.3-0.4)',
                'Adequate retinal vascularization pattern',
                'No evidence of increased intracranial pressure',
                'Clear optic disc boundaries',
                'Normal retinal pigmentation'
            ]
        
        # Select 3-5 features consistently
        num_features = 3 + (hash_int % 3)  # 3-5 features
        selected_features = all_features[:num_features]  # Take first N features consistently
        
        # Reset random seed to avoid affecting other random operations
        import time
        random.seed(int(time.time()))
        
        return {
            'is_tumor': is_tumor,
            'confidence': confidence,
            'features': selected_features,
            'prediction': 'Brain Tumor Indicators Detected' if is_tumor else 'Normal - No Tumor Indicators',
            'image_hash': image_hash[:16],  # Store hash for verification
            'analysis_method': 'Deterministic simulation based on image content'
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Fallback to basic random if image analysis fails
        return {
            'is_tumor': False,
            'confidence': 85.0,
            'features': ['Analysis incomplete - using fallback prediction'],
            'prediction': 'Normal - No Tumor Indicators',
            'image_hash': 'unknown',
            'analysis_method': 'Fallback prediction due to analysis error'
        }

def preprocess_image_pil(image_path):
    """Image preprocessing using PIL instead of OpenCV"""
    try:
        # Open and validate image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to standard size (512x512)
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Enhance contrast (similar to CLAHE)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Increase contrast
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            # Apply slight gaussian blur to reduce noise
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Save processed image (optional - for demonstration)
            processed_path = image_path.replace('.', '_processed.')
            img.save(processed_path, 'JPEG', quality=95)
            
            return True
            
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return False

def validate_medical_image(image_path):
    """Validate if uploaded image is suitable for medical analysis, including a check for fundus characteristics."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # 1. Check minimum resolution
            if width < 200 or height < 200:
                return False, "Image resolution too low. Minimum 200x200 pixels required."
            
            # 2. Check if image is too small in file size (might be corrupted)
            file_size = os.path.getsize(image_path)
            if file_size < 10000:  # Less than 10KB
                return False, "Image file too small. Please upload a proper retinal image."
            
            # 3. Check aspect ratio (should be reasonably square for retinal images)
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 3.0:
                return False, "Unusual image aspect ratio. Please upload a standard retinal image."
                
            # 4. Heuristic check for Retinal Fundus Image characteristics (Dark background with central content)
            
            # Convert to grayscale and numpy array for analysis
            gray_img = img.convert('L')
            pixels = np.array(gray_img)
            
            # Define central region (e.g., 50% of the image)
            h, w = pixels.shape
            center_h, center_w = h // 2, w // 2
            
            # Define a central mask (e.g., circle or square)
            # Using a central square region for simplicity: 25% area (50% side length)
            crop_size_h = h // 4 # 25% from center
            crop_size_w = w // 4
            
            center_region = pixels[center_h - crop_size_h : center_h + crop_size_h, 
                                   center_w - crop_size_w : center_w + crop_size_w]
                                   
            # Define peripheral region (e.g., excluding the inner 70% boundary)
            # Create a simple border mask (outer 15% boundary)
            border_h = h // 8
            border_w = w // 8
            
            peripheral_region = np.concatenate([
                pixels[:border_h, :].flatten(), # Top border
                pixels[-border_h:, :].flatten(), # Bottom border
                pixels[:, :border_w].flatten(), # Left border
                pixels[:, -border_w:].flatten() # Right border
            ])
            
            # Calculate mean brightness for center and periphery
            mean_center = np.mean(center_region)
            mean_peripheral = np.mean(peripheral_region)

            # Fundus image heuristic: The peripheral region (background) should be significantly darker
            # than the central region (the retina).
            # Fundus images usually have a dark background (low mean brightness).
            
            # Thresholds:
            # 1. Peripheral darkness: Mean peripheral brightness must be less than 80 (out of 255)
            # 2. Central contrast: Center must be brighter than periphery by at least 25 brightness levels
            
            is_dark_background = mean_peripheral < 80
            has_central_contrast = mean_center > (mean_peripheral + 25)

            if not is_dark_background:
                return False, f"Image failed background check. Peripheral brightness ({mean_peripheral:.1f}) suggests a light or complex background, not a standard fundus image."
            
            if not has_central_contrast:
                return False, f"Image failed contrast check. Central region ({mean_center:.1f}) is not significantly brighter than the periphery, suggesting a uniform or non-medical image."

            # If all checks pass
            return True, "Image validation successful"
            
    except Exception as e:
        # Handle cases where the image file might be corrupted or numpy/PIL fails unexpectedly
        return False, f"Image processing failed during validation: {str(e)}"

@app.route('/')
def index():
    # If a user ID is present in the session, they are logged in.
    if 'user_id' in session:
        # If the user is logged in, redirect them to the operational home page
        return redirect(url_for('home'))
        
    # If no user ID is present, ensure the session is cleared (just in case)
    # and show the landing page with Login/Register options.
    session.clear() # Added this to forcibly clear any lingering session data
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name'].strip()
        username = request.form['username'].strip().lower()
        email = request.form['email'].strip().lower()
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation
        if not all([full_name, username, email, password, confirm_password]):
            flash('Please fill in all fields.', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('register.html')
        
        conn = get_db_connection()
        
        # Check if username or email already exists
        existing_user = conn.execute(
            'SELECT id FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()
        
        if existing_user:
            flash('Username or email already exists.', 'error')
            conn.close()
            return render_template('register.html')
        
        # Create new user
        password_hash = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (full_name, username, email, password_hash) VALUES (?, ?, ?, ?)',
            (full_name, username, email, password_hash)
        )
        conn.commit()
        conn.close()
        
        flash('Registration successful! You can now login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip().lower()
        password = request.form['password']
        
        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')
        
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            flash(f'Welcome back, {user["full_name"]}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index')) # Redirect to the index/landing page after logout

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')
    
@app.route('/home')
def home():
    if 'user_id' not in session:
        # If they try to access /home without logging in, send them to the login screen
        return redirect(url_for('login')) 
    return render_template('home.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or BMP files.'}), 400
        
        # Secure filename
        filename = secure_filename(file.filename)
        if not filename:
            filename = 'uploaded_image' + file_ext
        
        # Add timestamp to filename to avoid conflicts
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{int(datetime.now().timestamp())}{ext}"
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(filepath)
        
        # Verify file was saved and validate it
        if not os.path.exists(filepath):
            return jsonify({'error': 'Failed to save file'}), 500
        
        # Validate medical image
        is_valid, validation_message = validate_medical_image(filepath)
        if not is_valid:
            # Remove invalid file
            os.remove(filepath)
            # Flash error message to be displayed on the client side
            flash(f'Upload rejected: {validation_message}', 'error')
            return jsonify({'success': False, 'error': validation_message}), 400
        
        # Store image info in session for analysis
        session['uploaded_image'] = {
            'filename': filename,
            'filepath': filepath,
            'original_name': file.filename,
            'file_size': os.path.getsize(filepath),
            'upload_time': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Image uploaded and validated successfully',
            'file_size': os.path.getsize(filepath)
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")  # For debugging
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/analysis')
def analysis():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if 'uploaded_image' not in session:
        flash('Please upload an image first.', 'error')
        return redirect(url_for('home'))
    
    return render_template('analysis.html', image_info=session['uploaded_image'])

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'uploaded_image' not in session:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_info = session['uploaded_image']
    filepath = image_info['filepath']
    
    try:
        # Step 1: Validate image still exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image file not found'}), 404
        
        # Step 2: Preprocess image using PIL
        preprocessing_success = preprocess_image_pil(filepath)
        if not preprocessing_success:
            return jsonify({'error': 'Image preprocessing failed'}), 500
        
        # Step 3: Extract features and Step 4: CNN prediction simulation
        prediction_result = simulate_cnn_prediction(filepath)
        
        # Step 5: Generate patient ID and prepare for database storage
        patient_id = f"{session['username'].upper()}_{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate processing time simulation
        processing_time = round(random.uniform(2.1, 5.8), 1)
        
        analysis_result = {
            'patient_id': patient_id,
            'image_name': image_info['original_name'],
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'features': prediction_result['features'],
            'is_tumor': prediction_result['is_tumor'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': processing_time,
            'image_dimensions': get_image_dimensions(filepath),
            'file_size_mb': round(image_info['file_size'] / (1024 * 1024), 2),
            'image_hash': prediction_result.get('image_hash', 'unknown'),
            'analysis_method': prediction_result.get('analysis_method', 'Standard CNN prediction')
        }
        
        # Store result in session for display
        session['analysis_result'] = analysis_result
        
        return jsonify({
            'success': True,
            'result': analysis_result
        })
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

def get_image_dimensions(image_path):
    """Get image dimensions using PIL"""
    try:
        with Image.open(image_path) as img:
            return f"{img.width}x{img.height}"
    except:
        return "Unknown"

@app.route('/save_result', methods=['POST'])
def save_result():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'analysis_result' not in session:
        return jsonify({'error': 'No analysis result to save'}), 400
    
    try:
        result = session['analysis_result']
        image_info = session['uploaded_image']
        
        print(f"Saving result for user {session['user_id']}: {result['patient_id']}")  # Debug log
        
        conn = get_db_connection()
        
        # Insert the analysis result
        cursor = conn.execute('''
            INSERT INTO analysis_results 
            (user_id, patient_id, image_name, image_path, prediction, confidence, features, is_tumor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session['user_id'],
            result['patient_id'],
            result['image_name'],
            image_info['filepath'],
            result['prediction'],
            result['confidence'],
            json.dumps(result['features']),
            1 if result['is_tumor'] else 0  # Ensure boolean is converted to int
        ))
        
        # Get the inserted record ID
        inserted_id = cursor.lastrowid
        print(f"Inserted record with ID: {inserted_id}")  # Debug log
        
        conn.commit()
        conn.close()
        
        if inserted_id:
            flash('Analysis result saved successfully!', 'success')
            return jsonify({
                'success': True, 
                'message': 'Result saved to database',
                'record_id': inserted_id
            })
        else:
            return jsonify({'error': 'Failed to insert record'}), 500
        
    except Exception as e:
        print(f"Save result error: {str(e)}")  # Debug log
        import traceback
        traceback.print_exc()  # Full error traceback
        return jsonify({'error': f'Failed to save result: {str(e)}'}), 500

@app.route('/database')
def database():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        conn = get_db_connection()
        
        # Debug: Check current user info
        current_user = conn.execute('SELECT id, username, full_name FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        print(f"Current user: ID={current_user['id'] if current_user else 'None'}, Username={current_user['username'] if current_user else 'None'}")
        
        # Debug: Check if table exists and has data
        table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results'").fetchone()
        if not table_check:
            print("ERROR: analysis_results table does not exist!")
            init_db()  # Reinitialize database
        
        # Count ALL records in the table
        all_count = conn.execute('SELECT COUNT(*) as count FROM analysis_results').fetchone()
        total_all_records = all_count['count'] if all_count else 0
        print(f"Total records in analysis_results table: {total_all_records}")
        
        # Count records for this specific user
        user_count = conn.execute('SELECT COUNT(*) as count FROM analysis_results WHERE user_id = ?', (session['user_id'],)).fetchone()
        total_user_records = user_count['count'] if user_count else 0
        print(f"Total records for user {session['user_id']}: {total_user_records}")
        
        # Debug: Show all records with user info
        if total_all_records > 0:
            all_records = conn.execute('''
                SELECT ar.id, ar.user_id, ar.patient_id, ar.prediction, u.username 
                FROM analysis_results ar 
                LEFT JOIN users u ON ar.user_id = u.id 
                ORDER BY ar.created_at DESC
            ''').fetchall()
            
            print("All records in database:")
            for record in all_records:
                print(f"  Record ID={record['id']}, User ID={record['user_id']}, Username={record['username'] if record['username'] else 'NULL'}, Patient={record['patient_id']}")
        
        # Get results for current user with JOIN to verify user exists
        results = conn.execute('''
            SELECT ar.patient_id, ar.image_name, ar.prediction, ar.confidence, 
                   ar.features, ar.is_tumor, ar.created_at, u.username, u.full_name
            FROM analysis_results ar
            JOIN users u ON ar.user_id = u.id
            WHERE ar.user_id = ?
            ORDER BY ar.created_at DESC
        ''', (session['user_id'],)).fetchall()
        
        conn.close()
        
        print(f"Retrieved {len(results)} results from database for current user")
        
        # Parse JSON features for display
        parsed_results = []
        for result in results:
            try:
                result_dict = dict(result)
                result_dict['features'] = json.loads(result_dict['features'])
                print(f"Processing result: {result_dict['patient_id']} - {result_dict['prediction']}")
                parsed_results.append(result_dict)
            except json.JSONDecodeError as e:
                print(f"JSON decode error for result: {e}")
                # Skip malformed records
                continue
        
        return render_template('database.html', results=parsed_results, current_user=current_user)
        
    except Exception as e:
        print(f"Database error: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('Error loading database results.', 'error')
        return render_template('database.html', results=[], current_user=None)

@app.route('/new_analysis')
def new_analysis():
    # Clear session data for new analysis
    if 'uploaded_image' in session:
        del session['uploaded_image']
    if 'analysis_result' in session:
        del session['analysis_result']
    
    return redirect(url_for('home'))

# Add a simple health check route
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    })

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files from the uploads directory"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Security check - ensure the file belongs to current session
    if 'uploaded_image' in session and session['uploaded_image']['filename'] == filename:
        return app.send_static_file(f'uploads/{filename}')
    else:
        # Check if file exists in database for this user
        conn = get_db_connection()
        result = conn.execute('''
            SELECT image_path FROM analysis_results 
            WHERE user_id = ? AND image_path LIKE ?
        ''', (session['user_id'], f'%{filename}')).fetchone()
        conn.close()
        
        if result:
            return app.send_static_file(f'uploads/{filename}')
    
    return "File not found or access denied", 404

if __name__ == '__main__':
    init_db()
    print("🚀 Starting Brain Tumor Detection System...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"💾 Database will be created at: brain_tumor_detection.db")
    print(f"🌐 Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
