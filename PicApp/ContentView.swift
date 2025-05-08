import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                NavigationLink(destination: DenoisingView()) {
                    FunctionalityButton(
                        imageName: "image-denoising",
                        title: "Image Denoising",
                        description: "Reduce image noise using state-of-the-art ML models."
                    )
                }

                NavigationLink(destination: ColorizerView()) {
                    FunctionalityButton(
                        imageName: "image-colorizer",
                        title: "Colorization",
                        description: "Bring back to life your old images."
                    )
                }
                Spacer()

                VStack(spacing: 4) {
                    Text("© 2025 Banaru Mircea-Sebastian. Developed as part of the Master's thesis at the University Politehnica of Bucharest.")
                        .font(.footnote)
                        .foregroundColor(.gray)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)

                    Text("Version \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0.0")")
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
                .padding(.bottom, 10)

            }
            .padding()
            .navigationTitle("Choose AI edit")
        }
    }
}
    
struct FunctionalityButton: View {
    let imageName: String
    let title: String
    let description: String
    
    var body: some View {
        HStack {
            Image(imageName)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 50, height: 50)
                .foregroundColor(.blue)
                .padding()
            
            VStack(alignment: .leading) {
                Text(title)
                    .font(.headline)
                    .fontWeight(.bold)
                Text(description)
                    .font(.subheadline)
                    .foregroundColor(.gray)
                    .multilineTextAlignment(.leading)
            }
            .padding(.leading, 0.0)
            
            Spacer()
        }
        .frame(maxWidth: .infinity, minHeight: 80)
        .background(Color.white)
        .cornerRadius(10)
        .shadow(color: .gray.opacity(0.3), radius: 5, x: 0, y: 2)
        .padding(.horizontal)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
