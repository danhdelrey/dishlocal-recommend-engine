import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from lightfm import LightFM
from lightfm.data import Dataset
import math

# =================================================================================
# SCRIPT HUẤN LUYỆN MODEL GỢI Ý - PHIÊN BẢN V5.0 (TIME-AWARE & RE-RANKING)
# - Hiểu và gán trọng số cho thời gian xem (view_duration_ms).
# - Đẩy các món đã tương tác xuống cuối danh sách thay vì loại bỏ.
# =================================================================================


# --- PHẦN 1: KẾT NỐI VÀ TRÍCH XUẤT DỮ LIỆU (EXTRACT) ---
def get_supabase_client() -> Client:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in .env file")
    return create_client(url, key)

def fetch_data(supabase: Client):
    print("Fetching all data from Supabase...")
    
    # Lấy dữ liệu chính
    all_users_res = supabase.table('profiles').select('id').execute()
    all_posts_res = supabase.table('posts').select('id, food_category').execute()
    
    # --- NÂNG CẤP: Lấy thêm cả cột view_duration_ms ---
    views_res = supabase.table('post_views').select('user_id, post_id, view_duration_ms').execute()
    likes_res = supabase.table('post_likes').select('user_id, post_id').execute()
    saves_res = supabase.table('post_saves').select('user_id, post_id').execute()
    explicit_reviews_res = supabase.table('post_reviews').select('post_id, rating').filter('rating', 'gte', 4).execute()
    review_choices_res = supabase.table('post_reviews').select('post_id, category, selected_choices').execute()

    # Xử lý Dữ liệu chính
    all_users_df = pd.DataFrame(all_users_res.data).rename(columns={'id': 'user_id'})
    all_posts_df = pd.DataFrame(all_posts_res.data).rename(columns={'id': 'post_id'})

    # --- Xử lý Tương tác với logic gán trọng số mới ---
    interaction_dfs = []

    # 1. Xử lý Lượt xem (Time-aware)
    views_df = pd.DataFrame(views_res.data)
    if not views_df.empty:
        def calculate_view_weight(duration_ms):
            if pd.isna(duration_ms) or duration_ms <= 0:
                return 1.0  # Impression, tín hiệu yếu
            # Dùng log để trọng số không tăng quá nhanh.
            # 1.5 + log(thời gian xem (s) + 1)
            return 1.5 + math.log((duration_ms / 1000) + 1)
        
        views_df['weight'] = views_df['view_duration_ms'].apply(calculate_view_weight)
        interaction_dfs.append(views_df[['user_id', 'post_id', 'weight']])

    # 2. Xử lý các tương tác tường minh khác
    likes_df = pd.DataFrame(likes_res.data)
    if not likes_df.empty:
        likes_df['weight'] = 4.0
        interaction_dfs.append(likes_df)

    saves_df = pd.DataFrame(saves_res.data)
    if not saves_df.empty:
        saves_df['weight'] = 5.0
        interaction_dfs.append(saves_df)
    
    reviews_df = pd.DataFrame(explicit_reviews_res.data)
    posts_authors_df = pd.DataFrame(supabase.table('posts').select('id, author_id').execute().data).rename(columns={'id': 'post_id', 'author_id': 'user_id'})
    if not reviews_df.empty and not posts_authors_df.empty:
        reviews_df = pd.merge(reviews_df, posts_authors_df, on='post_id')
        reviews_df['weight'] = 7.0 # Tín hiệu mạnh nhất
        interaction_dfs.append(reviews_df[['user_id', 'post_id', 'weight']])
    
    # Gộp tất cả các tương tác và lấy trọng số cao nhất cho mỗi cặp (user, post)
    if not interaction_dfs:
        interactions_df = pd.DataFrame(columns=['user_id', 'post_id', 'weight'])
    else:
        interactions_df = pd.concat(interaction_dfs)
        interactions_df = interactions_df.groupby(['user_id', 'post_id'])['weight'].max().reset_index()


    # --- Xử lý Đặc trưng Món ăn (Item Features) ---
    print("Processing item features...")
    item_features_list = []
    
    # 1. Thêm food_category làm feature
    if not all_posts_df.empty:
        for _, row in all_posts_df.dropna(subset=['food_category']).iterrows():
            item_features_list.append({'post_id': row['post_id'], 'feature': f"category:{row['food_category']}"})
    
    # 2. Thêm review_choices làm feature
    review_choices_df = pd.DataFrame(review_choices_res.data)
    if not review_choices_df.empty:
        for _, row in review_choices_df.iterrows():
            review_category = row['category']
            selected_choices = row.get('selected_choices')
            if selected_choices and isinstance(selected_choices, list):
                for choice in selected_choices:
                    feature_string = f"{review_category}:{choice}"
                    item_features_list.append({'post_id': row['post_id'], 'feature': feature_string})
    
    item_features_df = pd.DataFrame(item_features_list)
    
    print(f"Fetched {len(interactions_df)} interactions, {len(item_features_df)} item features for {len(all_users_df)} users.")
    return interactions_df, item_features_df, all_users_df, all_posts_df

# --- PHẦN 2: HUẤN LUYỆN MODEL ---
def train_model(interactions_df, item_features_df, all_users_df, all_posts_df):
    print("Preparing data and training model...")
    dataset = Dataset()
    
    all_possible_features = []
    if not item_features_df.empty:
        all_possible_features = item_features_df['feature'].unique()
        
    dataset.fit(
        users=all_users_df['user_id'].unique(),
        items=all_posts_df['post_id'].unique(),
        item_features=all_possible_features
    )

    (interactions_matrix, weights_matrix) = dataset.build_interactions(
        (row['user_id'], row['post_id'], row['weight']) for _, row in interactions_df.iterrows()
    )
    
    item_features_matrix = None
    if not item_features_df.empty:
        features_grouped = item_features_df.groupby('post_id')['feature'].apply(list)
        item_features_iterable = ((post_id, features) for post_id, features in features_grouped.items())
        item_features_matrix = dataset.build_item_features(item_features_iterable, normalize=True)

    model = LightFM(loss='warp', no_components=30, learning_rate=0.05, random_state=42)
    model.fit(weights_matrix, item_features=item_features_matrix, epochs=20, verbose=True)
    print("DEBUG: Model training has finished successfully.")
    
    return model, dataset, item_features_matrix, interactions_matrix

# --- PHẦN 3: TẠO GỢI Ý ---
def generate_and_load(supabase: Client, model: LightFM, dataset: Dataset, item_features_matrix, interactions_matrix, all_users_df):
    print("Generating and loading recommendations...")
    user_id_map, _, item_id_map, _ = dataset.mapping()
    all_item_indices = np.arange(len(item_id_map))
    recommendations_to_upsert = []

    # Một giá trị âm lớn nhưng hữu hạn để đẩy các món đã xem xuống cuối
    REORDER_PENALTY = -999.0

    for user_id in all_users_df['user_id']:
        user_index = user_id_map.get(user_id)
        if user_index is None:
            continue

        scores = model.predict(user_index, all_item_indices, item_features=item_features_matrix)
        
        if user_index < interactions_matrix.shape[0]:
            known_positives_indices = interactions_matrix.tocsr()[user_index].indices
            
            # Gán một điểm phạt để đẩy xuống cuối thay vì loại bỏ
            scores[known_positives_indices] = REORDER_PENALTY

        top_indices = np.argsort(-scores)[:100]
        
        for item_index in top_indices:
            post_id = list(item_id_map.keys())[item_index]
            recommendations_to_upsert.append({
                'user_id': user_id, 
                'post_id': post_id, 
                'score': float(scores[item_index]), 
                'model_version': 'lightfm_v5.0_time_aware' # Cập nhật phiên bản model
            })

    if recommendations_to_upsert:
        print(f"Upserting {len(recommendations_to_upsert)} recommendations...")
        # Xóa các gợi ý cũ của model này trước khi thêm mới
        supabase.table('user_post_recommendations').delete().eq('model_version', 'lightfm_v5.0_time_aware').execute()
        # Dùng upsert để ghi dữ liệu mới
        supabase.table('user_post_recommendations').upsert(recommendations_to_upsert).execute()
        print("Upsert completed.")

# --- PHẦN 4: HÀM MAIN ĐỂ CHẠY ---
if __name__ == '__main__':
    supabase_client = get_supabase_client()
    interactions, item_features, all_users, all_posts = fetch_data(supabase_client)
    
    if not interactions.empty:
        trained_model, data_dataset, features_mat, interactions_mat = train_model(interactions, item_features, all_users, all_posts)
        generate_and_load(supabase_client, trained_model, data_dataset, features_mat, interactions_mat, all_users)
    else:
        print("No interaction data found. Skipping training.")