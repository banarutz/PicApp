//
//  ColorizerView.swift
//  PicApp
//
//  Created by Sebastian Banaru on 30.01.2025.
//


import SwiftUI
import CoreML
import PhotosUI
import Accelerate

struct ColorizerView: View {
    @State private var isProcessing = false
    @State private var resultImage: UIImage? = nil
    @State private var inputImage: UIImage? = nil
    @State private var pickerDelegate: PickerDelegate?
    
    var body: some View {
        VStack {
            Button(action: loadImage) {
                Text("Load Image")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding()
            
            if isProcessing {
                ProgressView("Colorizing...")
                    .padding()
            }
            
            if let resultImage = resultImage {
                Image(uiImage: resultImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(height: 300)
                    .padding()
            } else {
                Text("Colorized image will appear here.")
                    .foregroundColor(.gray)
            }
            
            Spacer()
        }
        .navigationTitle("Image Colorizer")
    }
    
    func loadImage() {
        let configuration = PHPickerConfiguration(photoLibrary: .shared())
        let picker = PHPickerViewController(configuration: configuration)
        
        let delegate = PickerDelegate { image in
            DispatchQueue.main.async {
                if let image = image {
                    self.inputImage = image
                    self.processImage()
                }
            }
        }
        
        self.pickerDelegate = delegate
        picker.delegate = delegate
        
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootViewController = windowScene.windows.first?.rootViewController {
            rootViewController.present(picker, animated: true)
        }
    }
    
    func processImage() {
        guard let inputImage = inputImage else { return }
        isProcessing = true
        
        // Dimensiunile așteptate de model
        let targetSize = CGSize(width: 178, height: 218)
        
        DispatchQueue.global(qos: .userInitiated).async {
            // Redimensionează imaginea
            guard let resizedImage = inputImage.resized(to: targetSize) else {
                DispatchQueue.main.async {
                    self.isProcessing = false
                }
                return
            }
            
            // Continuă cu procesarea imaginii redimensionate
            guard let labImage = resizedImage.toLab(),
                  let lChannel = labImage.getLChannel() else {
                DispatchQueue.main.async {
                    self.isProcessing = false
                }
                return
            }
            
            // Step 2: Prepare model input
            do {
                let model = try VGGNet(configuration: MLModelConfiguration())
                let input = try VGGNetInput(x: lChannel)
                
                // Step 3: Run prediction
                let prediction = try model.prediction(input: input)
                
                // Step 4: Process output
                let aChannel = prediction.A
                let bChannel = prediction.B
                
                guard let colorized = combineLAB(l: lChannel, a: aChannel, b: bChannel, originalSize: targetSize) else {
                    throw NSError(domain: "ColorizationError", code: 3, userInfo: nil)
                }
                
                DispatchQueue.main.async {
                    self.resultImage = colorized
                    self.isProcessing = false
                }
            } catch {
                print("Colorization failed: \(error)")
                DispatchQueue.main.async {
                    self.isProcessing = false
                }
            }
        }
    }
    
    func combineLAB(l: MLMultiArray, a: MLMultiArray, b: MLMultiArray, originalSize: CGSize) -> UIImage? {
        let width = Int(originalSize.width)
        let height = Int(originalSize.height)
        
        var labPixels = [UInt8](repeating: 0, count: width * height * 3)
        
        for y in 0..<height {
            for x in 0..<width {
                let lIndex = [0, y, x] as [NSNumber]
                let aIndex = [0, y, x] as [NSNumber]
                let bIndex = [0, y, x] as [NSNumber]
                
                let L = UInt8((l[lIndex].doubleValue * 255).clamped(to: 0...255))
                let A = UInt8((a[aIndex].doubleValue * 255).clamped(to: 0...255))
                let B = UInt8((b[bIndex].doubleValue * 255).clamped(to: 0...255))
                
                let pixelIndex = (y * width + x) * 3
                labPixels[pixelIndex] = L
                labPixels[pixelIndex + 1] = A
                labPixels[pixelIndex + 2] = B
            }
        }
        
        return convertLabToRGB(labPixels: labPixels, width: width, height: height)
    }
    
    func convertLabToRGB(labPixels: [UInt8], width: Int, height: Int) -> UIImage? {
        // Implementare conversie LAB->RGB folosind Accelerate
        // (Această parte necesită o implementare mai complexă)
        // Urmărește https://docs.opencv.org/ pentru conversia corectă LAB->RGB
        
        // Implementare simplificată pentru demo:
        var rgbPixels = [UInt8](repeating: 0, count: width * height * 4)
        
        for i in 0..<labPixels.count/3 {
            let L = Double(labPixels[i*3])
            let a = Double(labPixels[i*3+1])
            let b = Double(labPixels[i*3+2])
            
            // Conversie LAB->RGB simplificată (înlocuiește cu formula corectă)
            let R = UInt8((L + a).clamped(to: 0...255))
            let G = UInt8((L - b).clamped(to: 0...255))
            let B = UInt8(L.clamped(to: 0...255))
            
            rgbPixels[i*4] = R
            rgbPixels[i*4+1] = G
            rgbPixels[i*4+2] = B
            rgbPixels[i*4+3] = 255
        }
        
        return UIImage.fromByteArray(rgbPixels, width: width, height: height)
    }
}

// MARK: - Extensii pentru procesare imagine
extension UIImage {
    // Funcție pentru redimensionarea imaginii
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, self.scale)
        defer { UIGraphicsEndImageContext() }
        self.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    // Funcție pentru conversia RGB -> LAB
    func toLab() -> UIImage? {
        guard let cgImage = self.cgImage else { return nil }
        
        // 1. Extrage pixelii RGB
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerRow = width * 4
        var rgbPixels = [UInt8](repeating: 0, count: width * height * 4)
        
        guard let context = CGContext(
            data: &rgbPixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // 2. Conversie RGB -> LAB
        var labPixels = [UInt8](repeating: 0, count: width * height * 4)
        
        for i in 0..<width*height {
            let r = Double(rgbPixels[i*4]) / 255.0
            let g = Double(rgbPixels[i*4+1]) / 255.0
            let b = Double(rgbPixels[i*4+2]) / 255.0
            
            // Conversie RGB -> XYZ
            let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
            let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
            let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
            
            // Conversie XYZ -> LAB (D65 illuminant)
            let xn = 0.95047
            let yn = 1.0
            let zn = 1.08883
            
            func f(_ t: Double) -> Double {
                return t > 0.008856 ? pow(t, 1/3) : (7.787 * t) + (16/116)
            }
            
            let fx = f(x/xn)
            let fy = f(y/yn)
            let fz = f(z/zn)
            
            // Calcul componente LAB
            let L = (116 * fy) - 16
            let a = 500 * (fx - fy)
            let bVal = 200 * (fy - fz)
            
            // Scalare la 0-255
            labPixels[i*4] = UInt8((L * 2.55).clamped(to: 0...255))       // L (0-100 -> 0-255)
            labPixels[i*4+1] = UInt8((a + 128).clamped(to: 0...255))     // A (-128-127 -> 0-255)
            labPixels[i*4+2] = UInt8((bVal + 128).clamped(to: 0...255))  // B (-128-127 -> 0-255)
            labPixels[i*4+3] = 255 // Alpha
        }
        
        // 3. Creează imaginea LAB
        guard let provider = CGDataProvider(data: Data(bytes: labPixels, count: labPixels.count) as CFData),
              let labCGImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            return nil
        }
        
        return UIImage(cgImage: labCGImage)
    }
    
    // Funcție pentru extragerea canalului L din imaginea LAB
    func getLChannel() -> MLMultiArray? {
        guard let cgImage = self.cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        
        // Extrage canalul L (stocat în prima componentă a pixelilor LAB)
        var labPixels = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &labPixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        do {
            // Creează MLMultiArray cu forma [1, 1, height, width]
            let array = try MLMultiArray(shape: [1, 1, height as NSNumber, width as NSNumber], dataType: .double)
            
            for y in 0..<height {
                for x in 0..<width {
                    let index = y * width + x
                    let L = Double(labPixels[index * 4]) / 255.0 // Normalizează la [0,1]
                    array[[0, 0, y, x] as [NSNumber]] = NSNumber(value: L)
                }
            }
            
            return array
        } catch {
            print("Eroare la crearea MLMultiArray: \(error)")
            return nil
        }
    }
    
    // Funcție pentru crearea unei imagini dintr-un array de bytes
    static func fromByteArray(_ bytes: [UInt8], width: Int, height: Int) -> UIImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        // Verifică dacă numărul de bytes este suficient
        guard bytes.count == width * height * 4 else {
            print("Numărul de bytes nu corespunde dimensiunilor imaginii")
            return nil
        }
        
        // Creează un CGDataProvider din array-ul de bytes
        guard let provider = CGDataProvider(data: Data(bytes: bytes, count: bytes.count) as CFData) else {
            print("Nu s-a putut crea CGDataProvider")
            return nil
        }
        
        // Creează un CGImage din bytes
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8, // 8 biți per componentă (R, G, B, A)
            bitsPerPixel: 32,    // 32 de biți per pixel (4 componente * 8 biți)
            bytesPerRow: width * 4, // 4 bytes per pixel (R, G, B, A)
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            print("Nu s-a putut crea CGImage")
            return nil
        }
        
        // Creează o UIImage din CGImage
        return UIImage(cgImage: cgImage)
    }
}

// Extensie pentru a limita valorile într-un interval
extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        return min(max(self, limits.lowerBound), limits.upperBound)
    }
}

// MARK: - Model CoreML
class VGGNetInput: NSObject, MLFeatureProvider {
    let x: MLMultiArray // Folosește 'x' în loc de 'input'
    
    // Implementarea protocolului MLFeatureProvider
    var featureNames: Set<String> {
        return ["x"] // Specifică numele corect al feature-ului
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "x" { // Verifică numele feature-ului
            return MLFeatureValue(multiArray: x)
        }
        return nil
    }
    
    init(x: MLMultiArray) { // Folosește 'x' în loc de 'input'
        self.x = x
    }
}


class VGGNet {
    let model: MLModel
    
    init(configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        model = try MLModel(contentsOf: VGGNet.urlOfModelInThisBundle)
    }
    
    func prediction(input: VGGNetInput) throws -> VGGNetOutput {
        let features = try model.prediction(from: input)
        return VGGNetOutput(A: features.featureValue(for: "A")!.multiArrayValue!,
                            B: features.featureValue(for: "B")!.multiArrayValue!)
    }
    
    static let urlOfModelInThisBundle: URL = {
        guard let url = Bundle.main.url(forResource: "resnet_mse_16_exported", withExtension: "mlmodelc") else {
            fatalError("Modelul nu a fost găsit în bundle")
        }
        return url
    }()
}

class VGGNetOutput: NSObject, MLFeatureProvider {
    let A: MLMultiArray
    let B: MLMultiArray
    
    var featureNames: Set<String> {
        return ["A", "B"]
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "A":
            return MLFeatureValue(multiArray: A)
        case "B":
            return MLFeatureValue(multiArray: B)
        default:
            return nil
        }
    }
    
    init(A: MLMultiArray, B: MLMultiArray) {
        self.A = A
        self.B = B
    }
}
