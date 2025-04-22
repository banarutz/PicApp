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
import Accelerate.vImage
import UniformTypeIdentifiers
import CoreImage
import CoreVideo
import CoreImage.CIFilterBuiltins


// MARK: - Delegate pentru PhotoPicker
//class PickerDelegate: NSObject, PHPickerViewControllerDelegate {
//    var completion: (UIImage?) -> Void
//
//    init(completion: @escaping (UIImage?) -> Void) {
//        self.completion = completion
//        super.init()
//    }
//
//    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
//        picker.dismiss(animated: true)
//
//        guard let result = results.first else {
//            completion(nil)
//            return
//        }
//
//        result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] object, error in
//            guard let image = object as? UIImage, error == nil else {
//                DispatchQueue.main.async {
//                    self?.completion(nil)
//                }
//                return
//            }
//
//            DispatchQueue.main.async {
//                self?.completion(image)
//            }
//        }
//    }
//}

struct ColorizerView: View {
    @State private var isProcessing = false
    @State private var resultImage: UIImage? = nil
    @State private var inputImage: UIImage? = nil
    @State private var pickerDelegate: PickerDelegate?
    @State private var errorMessage: String? = nil

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

            if let errorMessage = errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .padding()
            }

            if let resultImage = resultImage {
                Image(uiImage: resultImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(height: 300)
                    .padding()
            } else if !isProcessing {
                Text("Colorized image will appear here.")
                    .foregroundColor(.gray)
            }

            Spacer()
        }
        .navigationTitle("Image Colorizer")
    }

    func loadImage() {
        var configuration = PHPickerConfiguration(photoLibrary: .shared())
        configuration.filter = .images
        configuration.selectionLimit = 1

        let picker = PHPickerViewController(configuration: configuration)

        let delegate = PickerDelegate { image in
            if let image = image {
                self.inputImage = image
                self.errorMessage = nil
                self.processImage()
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
        guard let inputImage = inputImage else {
            self.errorMessage = "No input image selected"
            return
        }

        isProcessing = true
        errorMessage = nil

        let targetSize = CGSize(width: 218, height: 178) // Dimensiunile așteptate de model

DispatchQueue.global(qos: .userInitiated).async {
    do {
    guard let resizedImage = inputImage.resized(to: targetSize) else {
        throw NSError(domain: "ColorizationError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to resize image"])
    }
    print("Dimensiuni imagine redimensionată: \(resizedImage.size)")

    guard let labImage = resizedImage.toLab() else {
        throw NSError(domain: "ColorizationError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to convert to Lab colorspace"])
    }
    if let labCGImage = labImage.cgImage {
        print("Dimensiuni labCGImage după toLab: \(labCGImage.width) x \(labCGImage.height)")
    }

    guard let lChannelDouble = labImage.getLChannel() else {
    throw NSError(domain: "ColorizationError", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to extract L channel"])
}
print("Verificare valori brute din getLChannel():")
let lWidth = lChannelDouble.shape[3].intValue
let lHeight = lChannelDouble.shape[2].intValue
for y in 0..<min(10, lHeight) {
    for x in 0..<min(10, lWidth) {
        let lValue = lChannelDouble[[NSNumber(value: 0), NSNumber(value: 0), NSNumber(value: y), NSNumber(value: x)]].doubleValue
        print("Pixel (\(x), \(y)): L=\(lValue)")
    }
}

let lChannel: MLMultiArray
if lChannelDouble.dataType != MLMultiArrayDataType.float32 {
    guard let lChannelFloat = try? MLMultiArray(shape: lChannelDouble.shape as [NSNumber], dataType: .float32) else {
        throw NSError(domain: "ColorizationError", code: 8, userInfo: [NSLocalizedDescriptionKey: "Failed to convert L channel to Float32"])
    }
    lChannel = lChannelFloat

    // Copiază valorile din lChannelDouble în lChannel
    for y in 0..<lHeight {
        for x in 0..<lWidth {
            let index = [NSNumber(value: 0), NSNumber(value: 0), NSNumber(value: y), NSNumber(value: x)]
            lChannel[index] = NSNumber(value: lChannelDouble[index].doubleValue)
        }
    }
    print("Valorile din lChannelDouble au fost copiate în lChannel.")
} else {
    lChannel = lChannelDouble
}

print("Verificare interval valori din lChannel:")
var minL = Double.greatestFiniteMagnitude
var maxL = -Double.greatestFiniteMagnitude
for y in 0..<lHeight {
    for x in 0..<lWidth {
        let index = [NSNumber(value: 0), NSNumber(value: 0), NSNumber(value: y), NSNumber(value: x)]
        let value = lChannel[index].doubleValue
        minL = min(minL, value)
        maxL = max(maxL, value)
    }
}
print("Interval valori L: min=\(minL), max=\(maxL)")

    print("Input shape: \(lChannel.shape), dataType: \(lChannel.dataType)")

    let config = MLModelConfiguration()
    config.computeUnits = .all

    guard let model = try? VGGNet(configuration: config) else {
        throw NSError(domain: "ColorizationError", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to initialize ML model"])
    }

    let input = try VGGNetInput(x: lChannel)
    let prediction = try model.prediction(input: input)
    print("Chei disponibile în output-ul modelului: \(prediction.featureNames)")

    guard let aChannelOutput = prediction.featureValue(for: "A")?.multiArrayValue else {
        throw NSError(domain: "ColorizationError", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to get A channel from model output"])
    }

    guard let bChannelOutput = prediction.featureValue(for: "B")?.multiArrayValue else {
        throw NSError(domain: "ColorizationError", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to get B channel from model output"])
    }

    print("Verificare valori brute din canalul L:")
    for y in 0..<min(10, lHeight) {
        for x in 0..<min(10, lWidth) {
            let lValue = lChannel[[NSNumber(value: 0), NSNumber(value: 0), NSNumber(value: y), NSNumber(value: x)]].doubleValue
            print("Pixel (\(x), \(y)): L=\(lValue)")
        }
    }

    // Verificare output model
    print("Verificare output model:")
    let width = aChannelOutput.shape[1].intValue
    let height = aChannelOutput.shape[0].intValue
    for i in 0..<100 { // Verificăm primii 10 pixeli
        let aValue = aChannelOutput[[NSNumber(value: i / width), NSNumber(value: i % width)]].doubleValue
        let bValue = bChannelOutput[[NSNumber(value: i / width), NSNumber(value: i % width)]].doubleValue
        print("Pixel \(i): A=\(aValue), B=\(bValue)")
    }

    // Continuă cu combinarea canalelor LAB
    DispatchQueue.main.async {
        guard let colorizedImage = self.combineLAB(l: lChannel, a: aChannelOutput, b: bChannelOutput, originalSize: inputImage.size) else {
            print("Eroare: Imaginea generată este nil.")
            self.errorMessage = "Failed to generate colorized image"
            self.isProcessing = false
            return
        }

        print("Imaginea colorată a fost generată cu succes.")
        self.resultImage = colorizedImage
        self.isProcessing = false
        print("Imaginea colorată a fost afișată în interfața utilizatorului.")
    }
} catch {
    print("Colorization failed: \(error.localizedDescription)")
    DispatchQueue.main.async {
        self.errorMessage = "Colorization failed: \(error.localizedDescription)"
        self.isProcessing = false
    }
}
}
    }

    func combineLAB(l: MLMultiArray, a: MLMultiArray, b: MLMultiArray, originalSize: CGSize) -> UIImage? {
    // Utilizează dimensiunile array-urilor
    let width = l.shape[3].intValue
    let height = l.shape[2].intValue

    print("Processing image with dimensions: \(width)x\(height)")
    print("L shape: \(l.shape), A shape: \(a.shape), B shape: \(b.shape)")

    // Verifică dimensiunile array-urilor
    guard l.shape.count == 4, a.shape.count == 2, b.shape.count == 2,
          l.shape[2].intValue == a.shape[0].intValue,
          l.shape[3].intValue == a.shape[1].intValue,
          a.shape[0].intValue == b.shape[0].intValue,
          a.shape[1].intValue == b.shape[1].intValue else {
        print("Eroare: Dimensiunile array-urilor nu sunt consistente.")
        return nil
    }

    var labPixels = [UInt8](repeating: 0, count: width * height * 3)

    func safeGetValue(from array: MLMultiArray, x: Int, y: Int) -> Double? {
        let rank = array.shape.count
        if rank == 4 {
            return array[[NSNumber(value: 0), NSNumber(value: 0), NSNumber(value: y), NSNumber(value: x)] as [NSNumber]].doubleValue
        } else if rank == 3 {
            return array[[NSNumber(value: 0), NSNumber(value: y), NSNumber(value: x)] as [NSNumber]].doubleValue
        } else if rank == 2 {
            return array[[NSNumber(value: y), NSNumber(value: x)] as [NSNumber]].doubleValue
        }
        return nil
    }

    for y in 0..<height {
    for x in 0..<width {
        guard let lValue = safeGetValue(from: l, x: x, y: y),
              let aValue = safeGetValue(from: a, x: x, y: y),
              let bValue = safeGetValue(from: b, x: x, y: y) else {
            print("Eroare: Nu s-au putut accesa valorile pentru pixelul (\(x), \(y))")
            continue
        }

        let scaledL = UInt8(max(0, min(255, lValue * 2.55)))
        let scaledA = UInt8(max(0, min(255, aValue + 128)))
        let scaledB = UInt8(max(0, min(255, bValue + 128)))

        print("Pixel (\(x), \(y)): Scaled L=\(scaledL), Scaled A=\(scaledA), Scaled B=\(scaledB)")

        let pixelIndex = (y * width + x) * 3
        labPixels[pixelIndex] = scaledL
        labPixels[pixelIndex + 1] = scaledA
        labPixels[pixelIndex + 2] = scaledB
    }
}

print("Pixelii LAB au fost procesați.")

return convertLabToRGB(labPixels: labPixels, width: width, height: height)
}

    func convertLabToRGB(labPixels: [UInt8], width: Int, height: Int) -> UIImage? {
    var rgbPixels = [UInt8](repeating: 0, count: width * height * 4)

    for i in 0..<(width * height) {
        let labIndex = i * 3
        let l = Double(labPixels[labIndex]) / 255.0 * 100.0
        let a = Double(labPixels[labIndex + 1]) - 128.0
        let b = Double(labPixels[labIndex + 2]) - 128.0

        // Conversia LAB -> XYZ
        var y = (l + 16.0) / 116.0
        var x = a / 500.0 + y
        var z = y - b / 200.0

        x = pow(x, 3.0) > 0.008856 ? pow(x, 3.0) : (x - 16.0 / 116.0) / 7.787
        y = pow(y, 3.0) > 0.008856 ? pow(y, 3.0) : (y - 16.0 / 116.0) / 7.787
        z = pow(z, 3.0) > 0.008856 ? pow(z, 3.0) : (z - 16.0 / 116.0) / 7.787

        x *= 95.047
        y *= 100.000
        z *= 108.883

        // Conversia XYZ -> RGB
        x /= 100.0
        y /= 100.0
        z /= 100.0

        var r = x * 3.2406 + y * -1.5372 + z * -0.4986
        var g = x * -0.9689 + y * 1.8758 + z * 0.0415
        var bl = x * 0.0557 + y * -0.2040 + z * 1.0570

        r = r > 0.0031308 ? 1.055 * pow(r, 1.0 / 2.4) - 0.055 : r * 12.92
        g = g > 0.0031308 ? 1.055 * pow(g, 1.0 / 2.4) - 0.055 : g * 12.92
        bl = bl > 0.0031308 ? 1.055 * pow(bl, 1.0 / 2.4) - 0.055 : bl * 12.92

        let red = UInt8(max(0, min(255, r * 255.0)))
        let green = UInt8(max(0, min(255, g * 255.0)))
        let blue = UInt8(max(0, min(255, bl * 255.0)))

        let rgbIndex = i * 4
        rgbPixels[rgbIndex] = red
        rgbPixels[rgbIndex + 1] = green
        rgbPixels[rgbIndex + 2] = blue
        rgbPixels[rgbIndex + 3] = 255 // Alpha
    }

    guard let provider = CGDataProvider(data: Data(rgbPixels) as CFData) else {
        print("Eroare: Nu s-a putut crea CGDataProvider pentru RGB.")
        return nil
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

    guard let cgImage = CGImage(width: width,
                                height: height,
                                bitsPerComponent: 8,
                                bitsPerPixel: 32,
                                bytesPerRow: width * 4,
                                space: colorSpace,
                                bitmapInfo: bitmapInfo,
                                provider: provider,
                                decode: nil,
                                shouldInterpolate: false,
                                intent: .defaultIntent) else {
        print("Eroare: Nu s-a putut crea CGImage pentru RGB.")
        return nil
    }

    return UIImage(cgImage: cgImage)
}

    // MARK: - Funcție regândită pentru redimensionarea MLMultiArray folosind Core Image
    func resizeMLMultiArray(multiArray: MLMultiArray, toWidth: Int, toHeight: Int) -> MLMultiArray? {
        guard multiArray.dataType == .float32 else {
            print("Eroare: MLMultiArray trebuie să fie de tip Float32 pentru redimensionare cu Core Image.")
            return nil
        }

        let sourceWidth = multiArray.shape.last!.intValue
        let sourceHeight = multiArray.shape[multiArray.shape.count - 2].intValue
        let channelCount = (multiArray.shape.count > 2) ? multiArray.shape[multiArray.shape.count - 3].intValue : 1

        guard channelCount == 1 else {
            print("Eroare: Așteptat un MLMultiArray cu 1 canal.")
            return nil
        }

        let data = Data(bytesNoCopy: multiArray.dataPointer, count: multiArray.count * MemoryLayout<Float>.stride, deallocator: .none)
        let ciImage = CIImage(bitmapData: data,
                                    bytesPerRow: sourceWidth * MemoryLayout<Float>.stride,
                                    size: CGSize(width: sourceWidth, height: sourceHeight),
                                    format: CIFormat.Rf,
                                    colorSpace: nil)

        let filter = CIFilter.lanczosScaleTransform()
        filter.inputImage = ciImage
        let scale = CGFloat(toWidth) / CGFloat(sourceWidth)
        filter.scale = Float(scale)
        filter.aspectRatio = Float((CGFloat(toHeight) / CGFloat(sourceHeight)) / scale)

        guard let outputCIImage = filter.outputImage else {
            print("Eroare la aplicarea filtrului de redimensionare Core Image.")
            return nil
        }

        let context = CIContext()
        let resizedWidth = toWidth
        let resizedHeight = toHeight
        let resizedBytesPerRow = resizedWidth * MemoryLayout<Float>.stride
        let resizedData = UnsafeMutableRawPointer.allocate(byteCount: resizedHeight * resizedBytesPerRow, alignment: MemoryLayout<Float>.alignment)

        context.render(outputCIImage, toBitmap: UnsafeMutableRawPointer(resizedData), rowBytes: resizedBytesPerRow, bounds: CGRect(origin: .zero, size: CGSize(width: resizedWidth, height: resizedHeight)), format: CIFormat.Rf, colorSpace: nil)

        do {
            let destinationArray = try MLMultiArray(shape: [NSNumber(value: 1), NSNumber(value: 1), NSNumber(value: resizedHeight), NSNumber(value: resizedWidth)], dataType: .float32)
            let destinationPointer = destinationArray.dataPointer.assumingMemoryBound(to: Float.self)
            let sourcePointer = resizedData.assumingMemoryBound(to: Float.self)

            // Copiere directă a datelor Float
            for i in 0..<(resizedWidth * resizedHeight) {
                destinationPointer[i] = sourcePointer[i]
            }

            resizedData.deallocate()
            return destinationArray
        } catch {
            resizedData.deallocate()
            print("Eroare la crearea MLMultiArray destinație: \(error)")
            return nil
        }
    }
}

// MARK: - Extensii pentru procesare imagine
extension UIImage {
    // Funcție pentru redimensionarea imaginii
    func resized(to size: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: size))
        }
    }

    // Funcție pentru conversia RGB -> LAB
    func toLab() -> UIImage? {
        guard let cgImage = self.cgImage else { return nil }
        let resizedWidth = Int(self.size.width)
        let resizedHeight = Int(self.size.height)
        let bytesPerRow = resizedWidth * 4
        var rgbPixels = [UInt8](repeating: 0, count: resizedWidth * resizedHeight * 4)

                                        guard let context = CGContext(
                                            data: &rgbPixels,
                                            width: resizedWidth,
                                            height: resizedHeight,
                                            bitsPerComponent: 8,
                                            bytesPerRow: bytesPerRow,
                                            space: CGColorSpaceCreateDeviceRGB(),
                                            bitmapInfo: CGImageAlphaInfo
                                                .noneSkipLast.rawValue
                                        ) else { return nil }

                                        context.draw(cgImage, in: CGRect(origin: .zero, size: self.size))

                                        var labPixels = [UInt8](repeating: 0, count: resizedWidth * resizedHeight * 4)

                                        for i in 0..<resizedWidth*resizedHeight {
                                            let r = Double(rgbPixels[i*4]) / 255.0
                                            let g = Double(rgbPixels[i*4+1]) / 255.0
                                            let b = Double(rgbPixels[i*4+2]) / 255.0

                                            let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
                                            let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
                                            let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

                                            let xn = 0.95047
                                            let yn = 1.0
                                            let zn = 1.08883

                                            func f(_ t: Double) -> Double {
                                                return t > 0.008856 ? pow(t, 1/3) : (7.787 * t) + (16/116)
                                            }

                                            let fx = f(x/xn)
                                            let fy = f(y/yn)
                                            let fz = f(z/zn)

                                            let L = (116 * fy) - 16
                                            let a = 500 * (fx - fy)
                                            let bVal = 200 * (fy - fz)

                                            let L_byte = UInt8((L / 100.0 * 255.0).clamped(to: 0...255))
                                            let a_byte = UInt8(((a + 128.0) / 255.0 * 255.0).clamped(to: 0...255))
                                            let b_byte = UInt8(((bVal + 128.0) / 255.0 * 255.0).clamped(to: 0...255))

                                            labPixels[i*4] = L_byte
                                            labPixels[i*4+1] = a_byte
                                            labPixels[i*4+2] = b_byte
                                            labPixels[i*4+3] = 255
                                        }

                                        guard let provider = CGDataProvider(data: Data(bytes: labPixels, count: labPixels.count) as CFData),
                                              let labCGImage = CGImage(
                                                  width: resizedWidth,
                                                  height: resizedHeight,
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
            guard let cgImage = self.cgImage else {
                print("Eroare: getLChannel a primit o imagine fără CGImage.")
                return nil
            }
            let width = cgImage.width
            let height = cgImage.height

            print("Dimensiuni în getLChannel: \(cgImage.width) x \(cgImage.height)")

            guard let dataProvider = cgImage.dataProvider, let labCFData = dataProvider.data else {
                print("Eroare la obținerea datelor CGImage în getLChannel.")
                return nil
            }

            let labData = labCFData as Data
            let bytesPerRow = cgImage.bytesPerRow

            do {
                // MODIFICĂ AICI FORMA, inversând width și height
                let array = try MLMultiArray(shape: [NSNumber(value: 1), NSNumber(value: 1), NSNumber(value: width), NSNumber(value: height)], dataType: .double)

                for y in 0..<height {
                    for x in 0..<width {
                        let pixelIndex = y * bytesPerRow + x * 4
                        if pixelIndex + 0 < labData.count {
                            let lValue = Double(labData[pixelIndex + 0]) / 255.0
                            array[[NSNumber(value: 0), NSNumber(value: 0), NSNumber(value: y), NSNumber(value: x)] as [NSNumber]] = NSNumber(value: lValue)
                            if lValue < 0.0 || lValue > 1.0 {
                                print("Valoare L invalidă: \(lValue)")
                            }
                        }
                    }
                }
                return array
            } catch {
                print("Eroare la crearea MLMultiArray pentru canalul L: \(error)")
                return nil
            }
        }
                                }

                                // Extensia pentru convertirea Data la un array rămâne aceeași:
                                extension Data {
                                    func toArray<T>(type: T.Type) -> [T] where T: ExpressibleByIntegerLiteral {
                                        let count = self.count / MemoryLayout<T>.size
                                        var array = [T](repeating: 0, count: count)

                                        self.withUnsafeBytes { dataBuffer in
                                            guard let dataPtr = dataBuffer.baseAddress else {
                                                return
                                            }
                                            array.withUnsafeMutableBytes { arrayBuffer in
                                                guard let arrayPtr = arrayBuffer.baseAddress else {
                                                    return
                                                }
                                                memcpy(arrayPtr, dataPtr, count * MemoryLayout<T>.size)
                                            }
                                        }
                                        return array
                                    }
                                }

                                // Funcție pentru crearea unei imagini dintr-un array de bytes
                                extension UIImage {
                                    static func fromByteArray(_ bytes: [UInt8], width: Int, height: Int) -> UIImage? {
                                        let colorSpace = CGColorSpaceCreateDeviceRGB()
                                        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

                                        guard bytes.count == width * height * 4 else {
                                            print("Numărul de bytes nu corespunde dimensiunilor imaginii")
                                            return nil
                                        }

                                        var bytesArray = bytes

                                        guard let provider = CGDataProvider(data: Data(bytes: &bytesArray, count: bytesArray.count) as CFData) else {
                                            print("Nu s-a putut crea CGDataProvider")
                                            return nil
                                        }

                                        guard let cgImage = CGImage(
                                            width: width,
                                            height: height,
                                            bitsPerComponent: 8,
                                            bitsPerPixel: 32,
                                            bytesPerRow: width * 4,
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
                                    let x: MLMultiArray

                                    var featureNames: Set<String> {
                                        return ["input"]
                                    }

                                    func featureValue(for featureName: String) -> MLFeatureValue? {
                                        if featureName == "input" {
                                            return MLFeatureValue(multiArray: x)
                                        }
                                        return nil
                                    }

                                    init(x: MLMultiArray) throws {
                                        self.x = x
                                        super.init()
                                    }
                                }

                                class VGGNet {
                                    let model: MLModel

                                    init(configuration: MLModelConfiguration = MLModelConfiguration()) throws {
                                        guard let modelURL = Bundle.main.url(forResource: "VGG16_e50", withExtension: "mlmodelc") else {
                                            throw NSError(domain: "VGGNetError", code: 404, userInfo: [NSLocalizedDescriptionKey: "Modelul nu a fost găsit în bundle"])
                                        }

                                        do {
                                            model = try MLModel(contentsOf: modelURL, configuration: configuration)
                                        } catch {
                                            throw NSError(domain: "VGGNetError", code: 500, userInfo: [NSLocalizedDescriptionKey: "Eroare la încărcarea modelului: \(error.localizedDescription)"])
                                        }
                                    }

                                    func prediction(input: VGGNetInput) throws -> VGGNetOutput {
                                        do {
                                            let features = try model.prediction(from: input)
                                            
                                            do {
                                                let output = try model.prediction(from: input)
                                                print("Model output: \(output)")
                                            } catch {
                                                print("Eroare la apelul modelului: \(error.localizedDescription)")
                                            }
                                            
                                            if let rawOutput = features.featureValue(for: "var_342")?.multiArrayValue {
                                                let channelCount = rawOutput.shape[1].intValue // Ar trebui să fie 2

                                                let aShape = [NSNumber(value: rawOutput.shape[2].intValue), NSNumber(value: rawOutput.shape[3].intValue)]
                                                let bShape = [NSNumber(value: rawOutput.shape[2].intValue), NSNumber(value: rawOutput.shape[3].intValue)]

                                                let aArray = try MLMultiArray(shape: aShape, dataType: .float32)
                                                let bArray = try MLMultiArray(shape: bShape, dataType: .float32)

                                                let height = rawOutput.shape[2].intValue
                                                let width = rawOutput.shape[3].intValue

                                                for channel in 0..<channelCount {
                                                    for h in 0..<height {
                                                        for w in 0..<width {
                                                            let value = rawOutput[[NSNumber(value: 0), NSNumber(value: channel), NSNumber(value: h), NSNumber(value: w)]] as! NSNumber
                                                            if channel == 0 {
                                                                aArray[[NSNumber(value: h), NSNumber(value: w)]] = value
                                                            } else if channel == 1 {
                                                                bArray[[NSNumber(value: h), NSNumber(value: w)]] = value
                                                            }
                                                        }
                                                    }
                                                }
                                                // Returnează canalele separate
                                                return VGGNetOutput(A: aArray, B: bArray)
                                            } else {
                                                throw NSError(domain: "VGGNetError", code: 500, userInfo: [NSLocalizedDescriptionKey: "Output-ul modelului este invalid."])
                                            }
                                        } catch {
                                            print("Eroare: Modelul nu a reușit să genereze o predicție. Detalii: \(error.localizedDescription)")
                                            throw error // Aruncă eroarea mai departe pentru a fi gestionată de apelant
                                        }
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
                                            super.init()
                                        }
                                    }
                                }


