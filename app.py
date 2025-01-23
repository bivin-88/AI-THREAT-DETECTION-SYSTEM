import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hashlib
import plotly.express as px




def hash_ip(x, mod_value=2**16):
    return int(hashlib.sha256(str(x).encode()).hexdigest(), 16) % mod_value

def process_data(df, scaler, training_columns):
    """
    Process data with exact same protocols as training
    """
    # Define the exact columns from training
    protocol_columns = [
        'Protocol_BROWSER', 'Protocol_CLDAP', 'Protocol_CVSPSERVER', 'Protocol_Chargen',
        'Protocol_DCERPC', 'Protocol_DNS', 'Protocol_DRDA', 'Protocol_EPM', 'Protocol_HTTP',
        'Protocol_HTTP/JSON', 'Protocol_HTTP/XML', 'Protocol_ICMP', 'Protocol_ICMPv6',
        'Protocol_IMAP', 'Protocol_KRB5', 'Protocol_LDAP', 'Protocol_LSARPC', 'Protocol_MDNS',
        'Protocol_MySQL', 'Protocol_NBNS', 'Protocol_NBSS', 'Protocol_NFS', 'Protocol_NTP',
        'Protocol_POP', 'Protocol_Portmap', 'Protocol_RDP', 'Protocol_RDPUDP',
        'Protocol_RPC_NETLOGON', 'Protocol_SMB', 'Protocol_SMB2', 'Protocol_SMTP',
        'Protocol_SSH', 'Protocol_SSHv1', 'Protocol_SSHv2', 'Protocol_SSLv2', 'Protocol_TCP',
        'Protocol_TLSv1.2', 'Protocol_TLSv1.3', 'Protocol_UDP'
    ]
    
    # Preprocessing
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Drop unnecessary columns
    if 'Info' in df.columns:
        df = df.drop('Info', axis=1)
    if 'No.' in df.columns:
        df = df.drop('No.', axis=1)
    
    # Initialize protocol columns with zeros
    result_df = pd.DataFrame()
    result_df['Time'] = df['Time']
    result_df['Source'] = df['Source'].apply(hash_ip)
    result_df['Destination'] = df['Destination'].apply(hash_ip)
    result_df['Length'] = df['Length']
    
    # Initialize all protocol columns with 0
    for col in protocol_columns:
        result_df[col] = 0
    
    # Set the appropriate protocol column to 1
    for idx, row in df.iterrows():
        protocol_col = f'Protocol_{row["Protocol"]}'
        if protocol_col in protocol_columns:
            result_df.at[idx, protocol_col] = 1
    
    # Scale numerical features
    numeric_columns = ['Time', 'Length']
    result_df[numeric_columns] = scaler.transform(df[numeric_columns])
    
    # Reindex to match the training columns (ensuring the same column structure)
    result_df = result_df.reindex(columns=training_columns, fill_value=0)
    
    return result_df


def main():
    st.title("Network Anomaly Detection Dashboard")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load model, scaler, and training columns
            model = joblib.load('isolation_forest_model.pkl')
            scaler = joblib.load('scaler.pkl')
            training_columns = joblib.load('training_columns.pkl')  # Load training columns
            
            # Read data
            df = pd.read_csv(uploaded_file)
            original_df = df.copy()
            
            st.write("Original data shape:", df.shape)
            
            # Process data
            processed_df = process_data(df, scaler, training_columns)
            st.write("Processed data shape:", processed_df.shape)
            
            # Make predictions
            predictions = model.predict(processed_df)
            
            # Calculate results
            n_anomalies = (predictions == -1).sum()
            n_normal = (predictions == 1).sum()
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(predictions))
            with col2:
                st.metric("Anomalies", n_anomalies)
            with col3:
                st.metric("Normal", n_normal)
            
            # Visualizations
            fig = px.pie(
                names=['Normal', 'Anomaly'],
                values=[n_normal, n_anomalies],
                title='Distribution of Normal vs Anomalous Traffic',
                color_discrete_sequence=['green', 'red']
            )
            st.plotly_chart(fig)
            
            # Add predictions to original dataframe
            original_df['Anomaly'] = predictions
            anomalies_df = original_df[original_df['Anomaly'] == -1]
            
            # Protocol distribution in anomalies
            if len(anomalies_df) > 0:
                protocol_dist = anomalies_df['Protocol'].value_counts()
                fig2 = px.bar(
                    x=protocol_dist.index,
                    y=protocol_dist.values,
                    title='Protocol Distribution in Anomalies',
                    labels={'x': 'Protocol', 'y': 'Count'}
                )
                st.plotly_chart(fig2)
            
            # Display anomalies
            st.subheader("Detected Anomalies")
            if len(anomalies_df) > 0:
                st.dataframe(anomalies_df)
                
                # Download option
                csv = anomalies_df.to_csv(index=False)
                st.download_button(
                    label="Download Anomalies CSV",
                    data=csv,
                    file_name="network_anomalies.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Error details:", str(e))

if __name__ == "__main__":
    main()

