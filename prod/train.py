import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from lightfm import LightFM
from lightfm.data import Dataset

# =================================================================================
# SCRIPT HUẤN LUYỆN MODEL GỢI Ý - PHIÊN BẢN HOÀN THIỆN (FUTURE-PROOF)
# - Tự động sử dụng dữ liệu từ post_views, tags, post_tags nếu có.
# - Hoạt động ổn định ngay cả khi các bảng trên trống.
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
    print("Fetching data from Supabase...")
    
    # Lấy tất cả user và post để đảm bảo dataset đầy đủ
    all_users_res = supabase.table('profiles').select('id').execute()
    all_users_df = pd.DataFrame(all_users_res.data).rename(columns={'id': 'user_id'})
    
    all_posts_res = supabase.table('posts').select('id, food_category').execute()
    all_posts_df = pd.DataFrame(all_posts_res.data).rename(columns={'id': 'post_id'})

    # Lấy các loại tương tác
    views_res = supabase.table('post_views').select('user_id, post_id').execute()
    likes_res = supabase.table('post_likes').select('user_id, post_id').execute()
    saves_res = supabase.table('post_saves').select('user_id, post_id').execute()
    reviews_res = supabase.table('post_reviews').select('post_id, rating').filter('rating', 'gte', 4).execute()
    
    # Lấy dữ liệu đặc trưng (features) nếu có
    tags_res = supabase.table('tags').select('id, tag_name').execute()
    post_tags_res = supabase.table('post_tags').select('post_id, tag_id').execute()

    # Xử lý tương tác và gán trọng số
    views_df = pd.DataFrame(views_res.data); views_df['weight'] = 1.0 # Tín hiệu yếu
    likes_df = pd.DataFrame(likes_res.data); likes_df['weight'] = 2.0
    saves_df = pd.DataFrame(saves_res.data); saves_df['weight'] = 3.0
    
    reviews_df = pd.DataFrame(reviews_res.data)
    posts_authors_df = pd.DataFrame(supabase.table('posts').select('id, author_id').execute().data).rename(columns={'id': 'post_id', 'author_id': 'user_id'})
    if not reviews_df.empty and not posts_authors_df.empty:
        reviews_df = pd.merge(reviews_df, posts_authors_df, on='post_id')
        reviews_df['weight'] = 5.0 # Tín hiệu mạnh nhất
        reviews_df = reviews_df[['user_id', 'post_id', 'weight']]
    else:
        reviews_df = pd.DataFrame(columns=['user_id', 'post_id', 'weight'])

    interactions_df = pd.concat([views_df, likes_df, saves_df, reviews_df])
    if not interactions_df.empty:
        interactions_df = interactions_df.sort_values('weight', ascending=False).drop_duplicates(['user_id', 'post_id'])

    # Xử lý đặc trưng món ăn (item features)
    # Kết hợp 'food_category' và các 'tags' thành một bộ đặc trưng duy nhất
    item_features_list = []
    # 1. Thêm food_category làm feature
    if not all_posts_df.empty:
        for _, row in all_posts_df.dropna(subset=['food_category']).iterrows():
            item_features_list.append({'post_id': row['post_id'], 'feature': f"category:{row['food_category']}"})
    
    # 2. Thêm các tags từ bảng tags nếu có
    tags_df = pd.DataFrame(tags_res.data).rename(columns={'id': 'tag_id'})
    post_tags_df = pd.DataFrame(post_tags_res.data)
    if not tags_df.empty and not post_tags_df.empty:
        full_tags_df = pd.merge(post_tags_df, tags_df, on='tag_id')
        for _, row in full_tags_df.iterrows():
            item_features_list.append({'post_id': row['post_id'], 'feature': f"tag:{row['tag_name']}"})
    
    item_features_df = pd.DataFrame(item_features_list)
    
    print(f"Fetched {len(interactions_df)} interactions, {len(item_features_df)} item features for {len(all_users_df)} users.")
    return interactions_df, item_features_df, all_users_df, all_posts_df

# --- PHẦN 2: CHUẨN BỊ DỮ LIỆU & HUẤN LUYỆN MODEL (TRANSFORM & TRAIN) ---
def train_model(interactions_df, item_features_df, all_users_df, all_posts_df):
    print("Preparing data and training model...")
    dataset = Dataset()
    
    # Fit dataset với TẤT CẢ user, item và feature có thể có
    all_possible_features = []
    if not item_features_df.empty:
        all_possible_features = item_features_df['feature'].unique()
        
    dataset.fit(
        users=all_users_df['user_id'].unique(),
        items=all_posts_df['post_id'].unique(),
        item_features=all_possible_features
    )

    # Xây dựng ma trận tương tác
    (interactions_matrix, weights_matrix) = dataset.build_interactions(
        (row['user_id'], row['post_id'], row['weight']) for index, row in interactions_df.iterrows()
    )
    
    # Xây dựng ma trận đặc trưng (nếu có dữ liệu)
    item_features_matrix = None
    if not item_features_df.empty:
        # Gom nhóm các feature cho mỗi post_id
        features_grouped = item_features_df.groupby('post_id')['feature'].apply(list)
        item_features_iterable = ((post_id, features) for post_id, features in features_grouped.items())
        item_features_matrix = dataset.build_item_features(item_features_iterable, normalize=True)

    # Huấn luyện model
    model = LightFM(loss='warp', no_components=30, learning_rate=0.05, random_state=42)
    model.fit(weights_matrix, item_features=item_features_matrix, epochs=20, verbose=True)
    print("DEBUG: Model training has finished successfully.")
    
    return model, dataset, item_features_matrix, interactions_matrix

# --- PHẦN 3: TẠO GỢI Ý VÀ LƯU VÀO DATABASE (LOAD) ---
def generate_and_load(supabase: Client, model: LightFM, dataset: Dataset, item_features_matrix, interactions_matrix, all_users_df):
    print("Generating and loading recommendations...")
    user_id_map, _, item_id_map, _ = dataset.mapping()
    all_item_indices = np.arange(len(item_id_map))
    recommendations_to_upsert = []

    for user_id in all_users_df['user_id']:
        user_index = user_id_map.get(user_id)
        if user_index is None:
            continue # Bỏ qua nếu user không có trong map (trường hợp hiếm)

        scores = model.predict(user_index, all_item_indices, item_features=item_features_matrix)
        
        # Lọc ra những món đã tương tác nếu người dùng có trong ma trận tương tác
        if user_index < interactions_matrix.shape[0]:
            known_positives_indices = interactions_matrix.tocsr()[user_index].indices
            scores[known_positives_indices] = -np.inf

        top_indices = np.argsort(-scores)[:100]
        
        for item_index in top_indices:
            if scores[item_index] > -np.inf:
                post_id = list(item_id_map.keys())[item_index]
                recommendations_to_upsert.append({
                    'user_id': user_id, 
                    'post_id': post_id, 
                    'score': float(scores[item_index]), 
                    'model_version': 'lightfm_v3.0_hybrid'
                })

    if recommendations_to_upsert:
        print(f"Upserting {len(recommendations_to_upsert)} recommendations...")
        supabase.table('user_post_recommendations').delete().eq('model_version', 'lightfm_v3.0_hybrid').execute()
        supabase.table('user_post_recommendations').upsert(recommendations_to_upsert).execute()
        print("Upsert completed.")

# --- HÀM MAIN ĐỂ CHẠY TOÀN BỘ QUY TRÌNH ---
if __name__ == '__main__':
    supabase_client = get_supabase_client()
    interactions, item_features, all_users, all_posts = fetch_data(supabase_client)
    
    # Chỉ chạy huấn luyện nếu có ít nhất một tương tác
    if not interactions.empty:
        trained_model, data_dataset, features_mat, interactions_mat = train_model(interactions, item_features, all_users, all_posts)
        generate_and_load(supabase_client, trained_model, data_dataset, features_mat, interactions_mat, all_users)
    else:
        print("No interaction data found. Skipping training.")